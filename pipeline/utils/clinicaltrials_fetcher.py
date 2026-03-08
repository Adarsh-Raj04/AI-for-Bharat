"""
ClinicalTrials.gov Fetcher - Fetches clinical trial data
"""
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ClinicalTrialsFetcher:
    """
    Fetches clinical trial data from ClinicalTrials.gov API
    API Docs: https://clinicaltrials.gov/data-api/api
    """
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2"
        self.rate_limit_delay = 1.0  # 1 request per second
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        status: Optional[str] = None,
        phase: Optional[str] = None
    ) -> List[Dict]:
        """
        Search clinical trials
        
        Args:
            query: Search query (condition, intervention, etc.)
            max_results: Maximum number of results
            status: Trial status (e.g., "RECRUITING", "COMPLETED")
            phase: Trial phase (e.g., "PHASE3")
            
        Returns:
            List of clinical trial details
        """
        trials = []
        page_size = 100
        
        for page in range(0, max_results, page_size):
            batch = self._fetch_page(query, page, min(page_size, max_results - page), status, phase)
            trials.extend(batch)
            
            if len(batch) < page_size:
                break
            
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched {len(trials)} clinical trials for query: {query}")
        return trials[:max_results]
    
    def _fetch_page(
        self,
        query: str,
        page: int,
        page_size: int,
        status: Optional[str],
        phase: Optional[str]
    ) -> List[Dict]:
        """Fetch a page of results"""
        params = {
            "query.term": query,
            "pageSize": page_size,
            "format": "json",
        }
        
        # Note: pageToken might not work as expected, using different approach
        if page > 0:
            params["pageToken"] = str(page)
        
        if status:
            params["filter.overallStatus"] = status
        if phase:
            params["filter.phase"] = phase
        
        try:
            url = f"{self.base_url}/studies"
            logger.info(f"Fetching from ClinicalTrials: {url} with params: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Received {len(studies)} studies from ClinicalTrials API")
            
            parsed_studies = []
            for study in studies:
                parsed = self._parse_study(study)
                if parsed and parsed.get("nct_id"):  # Only add if valid
                    parsed_studies.append(parsed)
            
            return parsed_studies
            
        except requests.exceptions.Timeout:
            logger.error(f"ClinicalTrials API timeout after 30 seconds")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"ClinicalTrials API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch clinical trials page: {e}")
            return []
    
    def _parse_study(self, study: Dict) -> Dict:
        """Parse a clinical trial study"""
        try:
            protocol = study.get("protocolSection", {})
            
            if not protocol:
                logger.warning("Study missing protocolSection")
                return {}
            
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            description = protocol.get("descriptionModule", {})
            conditions = protocol.get("conditionsModule", {})
            design = protocol.get("designModule", {})
            arms = protocol.get("armsInterventionsModule", {})
            outcomes = protocol.get("outcomesModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            contacts = protocol.get("contactsLocationsModule", {})
            sponsors = protocol.get("sponsorCollaboratorsModule", {})
            
            # NCT ID
            nct_id = identification.get("nctId", "")
            
            if not nct_id:
                logger.warning("Study missing NCT ID")
                return {}
            
            # Title
            title = identification.get("officialTitle") or identification.get("briefTitle", "")
            
            # Summary
            brief_summary = description.get("briefSummary", "")
            detailed_description = description.get("detailedDescription", "")
            summary = f"{brief_summary}\n\n{detailed_description}".strip()
            
            if not summary:
                logger.warning(f"Study {nct_id} has no summary")
                summary = title  # Use title as fallback
            
            # Conditions
            condition_list = conditions.get("conditions", [])
            
            # Phase
            phase_list = design.get("phases", [])
            phase = ", ".join(phase_list) if phase_list else "N/A"
            
            # Status
            overall_status = status.get("overallStatus", "")
            
            # Dates
            start_date = status.get("startDateStruct", {}).get("date")
            completion_date = status.get("completionDateStruct", {}).get("date")
            
            # Interventions
            interventions = arms.get("interventions", [])
            intervention_names = [i.get("name", "") for i in interventions]
            
            # Primary outcome
            primary_outcomes = outcomes.get("primaryOutcomes", [])
            primary_outcome = primary_outcomes[0].get("measure", "") if primary_outcomes else ""
            
            # Enrollment
            enrollment = design.get("enrollmentInfo", {}).get("count", 0)
            
            # Sponsor
            lead_sponsor = sponsors.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name", "")
            sponsor_class = lead_sponsor.get("class", "")
            
            # Check if industry-sponsored
            industry_sponsored = sponsor_class in ["INDUSTRY", "INDUSTRY_FED"]
            
            # Collaborators
            collaborators = sponsors.get("collaborators", [])
            collaborator_names = [c.get("name", "") for c in collaborators]
            
            # Locations
            locations = contacts.get("locations", [])
            countries = list(set([loc.get("country", "") for loc in locations if loc.get("country")]))
            
            return {
                "id": f"NCT:{nct_id}",
                "nct_id": nct_id,
                "title": title,
                "summary": summary,
                "conditions": condition_list,
                "phase": phase,
                "status": overall_status,
                "start_date": start_date,
                "completion_date": completion_date,
                "interventions": intervention_names,
                "primary_outcome": primary_outcome,
                "enrollment": enrollment,
                "sponsor": sponsor_name,
                "sponsor_class": sponsor_class,
                "collaborators": collaborator_names,
                "countries": countries,
                "source_type": "clinical_trial",
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "industry_sponsored": industry_sponsored,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to parse study: {e}")
            return {}
    
    def fetch_by_nct_id(self, nct_id: str) -> Optional[Dict]:
        """
        Fetch a specific trial by NCT ID
        
        Args:
            nct_id: NCT identifier (e.g., "NCT04280705")
            
        Returns:
            Trial details or None
        """
        try:
            response = requests.get(f"{self.base_url}/studies/{nct_id}")
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            if studies:
                return self._parse_study(studies[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch NCT {nct_id}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    fetcher = ClinicalTrialsFetcher()
    trials = fetcher.search("diabetes", max_results=10, phase="PHASE3")
    
    for trial in trials:
        print(f"NCT: {trial['nct_id']}")
        print(f"Title: {trial['title']}")
        print(f"Phase: {trial['phase']}")
        print(f"Status: {trial['status']}")
        print(f"Industry Sponsored: {trial['industry_sponsored']}")
        print("-" * 80)
