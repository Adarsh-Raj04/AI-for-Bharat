"""
bioRxiv/medRxiv Fetcher - Fetches preprints from bioRxiv and medRxiv
"""
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BioRxivFetcher:
    """
    Fetches preprints from bioRxiv and medRxiv
    API Docs: https://api.biorxiv.org/
    """
    
    def __init__(self, server: str = "medrxiv"):
        """
        Args:
            server: "biorxiv" or "medrxiv"
        """
        self.server = server
        self.base_url = f"https://api.biorxiv.org/details/{server}"
        self.rate_limit_delay = 1.0
    
    def fetch_by_date_range(
        self,
        start_date: str,
        end_date: str,
        cursor: int = 0,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Fetch preprints by date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cursor: Starting position
            max_results: Maximum number of results
            
        Returns:
            List of preprint details
        """
        preprints = []
        current_cursor = cursor
        
        # bioRxiv API returns max 100 per call, need pagination
        while len(preprints) < max_results:
            try:
                url = f"{self.base_url}/{start_date}/{end_date}/{current_cursor}"
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                collection = data.get("collection", [])
                
                if not collection:
                    break  # No more results
                
                for item in collection:
                    if len(preprints) >= max_results:
                        break
                    preprint = self._parse_preprint(item)
                    if preprint:
                        preprints.append(preprint)
                
                # If we got less than 100, we've reached the end
                if len(collection) < 100:
                    break
                
                current_cursor += 100
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch {self.server} preprints at cursor {current_cursor}: {e}")
                break
        
        logger.info(f"Fetched {len(preprints)} preprints from {self.server}")
        return preprints
    
    def fetch_recent(self, days: int = 7, max_results: int = 100) -> List[Dict]:
        """
        Fetch recent preprints
        
        Args:
            days: Number of days back to fetch
            max_results: Maximum number of results
            
        Returns:
            List of preprint details
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_by_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            max_results=max_results
        )
    
    def search_by_category(
        self,
        category: str,
        start_date: str,
        end_date: str,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search by subject category
        
        Args:
            category: Subject category (e.g., "infectious diseases")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum number of results
            
        Returns:
            List of preprint details
        """
        all_preprints = self.fetch_by_date_range(start_date, end_date, max_results=max_results*2)
        
        # Filter by category
        filtered = [
            p for p in all_preprints
            if category.lower() in p.get("category", "").lower()
        ]
        
        return filtered[:max_results]
    
    def _parse_preprint(self, item: Dict) -> Optional[Dict]:
        """Parse a preprint item"""
        try:
            doi = item.get("doi", "")
            
            if not doi:
                return None
            
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            
            # Authors
            authors_str = item.get("authors", "")
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]
            
            # Dates
            date = item.get("date", "")
            
            # Category
            category = item.get("category", "")
            
            # Version
            version = item.get("version", "1")
            
            # Check for industry sponsorship (preprints less likely but check)
            industry_sponsored = self._check_industry_sponsorship(abstract)
            
            return {
                "id": f"DOI:{doi}",
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": date,
                "category": category,
                "version": version,
                "source_type": self.server,
                "url": f"https://www.{self.server}.org/content/{doi}",
                "industry_sponsored": industry_sponsored,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to parse preprint: {e}")
            return None
    
    def _check_industry_sponsorship(self, abstract: str) -> bool:
        """Check if preprint is industry-sponsored"""
        keywords = [
            "sponsored by",
            "funded by",
            "pharmaceutical company",
            "industry-sponsored",
            "commercial sponsor",
        ]
        
        abstract_lower = abstract.lower()
        return any(keyword in abstract_lower for keyword in keywords)


# Example usage
if __name__ == "__main__":
    # Fetch from medRxiv
    fetcher = BioRxivFetcher(server="medrxiv")
    preprints = fetcher.fetch_recent(days=30, max_results=10)
    
    for preprint in preprints:
        print(f"DOI: {preprint['doi']}")
        print(f"Title: {preprint['title']}")
        print(f"Date: {preprint['publication_date']}")
        print(f"Category: {preprint['category']}")
        print("-" * 80)
