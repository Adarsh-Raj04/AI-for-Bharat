"""
PubMed Fetcher - Fetches research papers from PubMed E-utilities API
"""
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


class PubMedFetcher:
    """
    Fetches research papers from PubMed using E-utilities API
    API Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    
    def __init__(self, email: str = "your-email@example.com", api_key: Optional[str] = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email
        self.api_key = api_key
        self.rate_limit_delay = 0.34 if not api_key else 0.1  # 3/sec without key, 10/sec with key
    
    def search(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs
        
        Args:
            query: Search query (e.g., "diabetes treatment")
            max_results: Maximum number of results
            start_date: Start date (YYYY/MM/DD)
            end_date: End date (YYYY/MM/DD)
            
        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        if start_date:
            params["mindate"] = start_date
        if end_date:
            params["maxdate"] = end_date
        
        try:
            response = requests.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            logger.info(f"Found {len(pmids)} PMIDs for query: {query}")
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper details
        """
        if not pmids:
            return []
        
        papers = []
        
        # Fetch in batches of 200 (API limit)
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            batch_papers = self._fetch_batch(batch)
            papers.extend(batch_papers)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched details for {len(papers)} papers")
        return papers
    
    def _fetch_batch(self, pmids: List[str]) -> List[Dict]:
        """Fetch a batch of papers"""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(f"{self.base_url}/efetch.fcgi", params=params)
            response.raise_for_status()
            
            return self._parse_xml(response.text)
            
        except Exception as e:
            logger.error(f"Failed to fetch batch: {e}")
            return []
    
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
        
        return papers
    
    def _parse_article(self, article: ET.Element) -> Optional[Dict]:
        """Parse a single article"""
        try:
            # PMID
            pmid = article.find(".//PMID")
            pmid = pmid.text if pmid is not None else None
            
            if not pmid:
                return None
            
            # Title
            title = article.find(".//ArticleTitle")
            title = title.text if title is not None else ""
            
            # Abstract
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join([
                part.text for part in abstract_parts if part.text
            ])
            
            # Authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    authors.append(name)
            
            # Publication date
            pub_date = article.find(".//PubDate")
            year = pub_date.find("Year") if pub_date is not None else None
            month = pub_date.find("Month") if pub_date is not None else None
            day = pub_date.find("Day") if pub_date is not None else None
            
            pub_date_str = None
            if year is not None:
                pub_date_str = year.text
                if month is not None:
                    pub_date_str += f"-{month.text}"
                    if day is not None:
                        pub_date_str += f"-{day.text}"
            
            # Journal
            journal = article.find(".//Journal/Title")
            journal = journal.text if journal is not None else ""
            
            # DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # Check for industry sponsorship
            industry_sponsored = self._check_industry_sponsorship(abstract, article)
            
            return {
                "id": f"PMID:{pmid}",
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": pub_date_str,
                "journal": journal,
                "doi": doi,
                "source_type": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "industry_sponsored": industry_sponsored,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def _check_industry_sponsorship(self, abstract: str, article: ET.Element) -> bool:
        """Check if paper is industry-sponsored"""
        keywords = [
            "sponsored by",
            "funded by",
            "pharmaceutical company",
            "industry-sponsored",
            "commercial sponsor",
        ]
        
        # Check abstract
        abstract_lower = abstract.lower()
        for keyword in keywords:
            if keyword in abstract_lower:
                return True
        
        # Check grants
        for grant in article.findall(".//Grant"):
            agency = grant.find("Agency")
            if agency is not None and agency.text:
                agency_lower = agency.text.lower()
                if any(word in agency_lower for word in ["pharma", "inc", "corp", "ltd"]):
                    return True
        
        return False
    
    def fetch_by_query(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Search and fetch papers in one call
        
        Args:
            query: Search query
            max_results: Maximum number of results
            start_date: Start date (YYYY/MM/DD)
            end_date: End date (YYYY/MM/DD)
            
        Returns:
            List of paper details
        """
        pmids = self.search(query, max_results, start_date, end_date)
        return self.fetch_details(pmids)


# Example usage
if __name__ == "__main__":
    fetcher = PubMedFetcher(email="your-email@example.com")
    papers = fetcher.fetch_by_query("diabetes treatment", max_results=10)
    
    for paper in papers:
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Industry Sponsored: {paper['industry_sponsored']}")
        print("-" * 80)
