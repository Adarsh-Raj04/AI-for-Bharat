"""
URL Fetcher & Document Parser
Fetches and extracts clean text from any URL or uploaded PDF/DOCX.
"""

import io
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL Fetcher
# ---------------------------------------------------------------------------


class URLFetcher:
    """Fetches a URL and returns clean text + metadata."""

    # Real browser UA — PubMed and ClinicalTrials block bot UAs with 403
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # NCBI E-utilities base — free API, no key needed for low volume
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def fetch(self, url: str) -> dict:
        """
        Fetch URL and return structured result.

        Returns:
            {
                "text": str,
                "title": str,
                "url": str,
                "source_type": str,   # "pubmed" | "clinical_trial" | "biorxiv" | "web"
                "pmid": str | None,
                "nct_id": str | None,
                "error": str | None,
            }
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""

            # PubMed: use E-utilities API instead of scraping (avoids 403)
            if "pubmed.ncbi.nlm.nih.gov" in hostname:
                pmid = re.search(r"/(\d{7,9})/?", url)
                if pmid:
                    return self._fetch_pubmed_eutils(pmid.group(1), url)

            with httpx.Client(
                headers=self.HEADERS,
                follow_redirects=True,
                timeout=20.0,
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                html = resp.text

            # Route to specialised parser
            if "pubmed.ncbi.nlm.nih.gov" in hostname:
                return self._parse_pubmed(html, url)
            elif "clinicaltrials.gov" in hostname:
                return self._parse_clinicaltrials(html, url)
            elif "biorxiv.org" in hostname or "medrxiv.org" in hostname:
                return self._parse_biorxiv(html, url)
            else:
                return self._parse_generic(html, url)

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error fetching %s: %s", url, e)
            return self._error_result(url, f"HTTP {e.response.status_code}")
        except Exception as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return self._error_result(url, str(e))

    # ── Specialised parsers ───────────────────────────────────────────────

    def _fetch_pubmed_eutils(self, pmid: str, url: str) -> dict:
        """
        Fetch PubMed abstract via NCBI E-utilities API.
        Reliable, no scraping, no 403s.
        Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
        """
        try:
            with httpx.Client(headers=self.HEADERS, timeout=15.0) as client:
                # efetch returns XML with full abstract
                resp = client.get(
                    f"{self.EUTILS_BASE}/efetch.fcgi",
                    params={
                        "db": "pubmed",
                        "id": pmid,
                        "rettype": "abstract",
                        "retmode": "xml",
                    },
                )
                resp.raise_for_status()
                xml = resp.text

            soup = BeautifulSoup(xml, "xml")

            title = self._text(soup.find("ArticleTitle"))
            abstract_parts = soup.find_all("AbstractText")
            abstract = " ".join(
                (f"{a.get('Label', '')}: " if a.get("Label") else "")
                + a.get_text(" ", strip=True)
                for a in abstract_parts
            )

            # Authors
            authors = []
            for author in soup.find_all("Author")[:6]:
                last = self._text(author.find("LastName"))
                fore = self._text(author.find("ForeName"))
                if last:
                    authors.append(f"{last} {fore}".strip())
            authors_str = ", ".join(authors)

            journal = self._text(soup.find("ISOAbbreviation") or soup.find("Title"))
            pub_year = self._text(
                soup.find("PubDate").find("Year") if soup.find("PubDate") else None
            )

            text = f"{title}\n\n{abstract}".strip()
            if not text:
                # Fallback to HTML scraping if XML came back empty
                logger.warning(
                    "E-utilities returned empty content for PMID:%s, trying HTML", pmid
                )
                return self._fetch_pubmed_html(pmid, url)

            return {
                "text": text,
                "title": title,
                "url": url,
                "source_type": "pubmed",
                "pmid": pmid,
                "nct_id": None,
                "authors": authors_str,
                "journal": journal,
                "publication_date": pub_year,
                "error": None,
            }

        except Exception as e:
            logger.warning(
                "E-utilities failed for PMID:%s (%s), trying HTML fallback", pmid, e
            )
            return self._fetch_pubmed_html(pmid, url)

    def _fetch_pubmed_html(self, pmid: str, url: str) -> dict:
        """HTML scrape fallback for PubMed (used only if E-utilities fails)."""
        try:
            with httpx.Client(
                headers=self.HEADERS, follow_redirects=True, timeout=20.0
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
            return self._parse_pubmed(resp.text, url)
        except Exception as e:
            return self._error_result(url, f"PubMed fetch failed: {e}")

    def _parse_pubmed(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        # PMID from URL
        pmid = re.search(r"/(\d{7,9})/?", url)
        pmid = pmid.group(1) if pmid else None

        title = self._text(soup.select_one("h1.heading-title"))
        abstract_el = soup.select_one("#abstract")
        abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""

        # Authors, journal, date
        authors_el = soup.select(".authors-list .author-name")
        authors = ", ".join(a.get_text(strip=True) for a in authors_el[:6])
        journal = self._text(
            soup.select_one(".journal-actions .NLM_abbrev-journal-title")
        )
        pub_date = self._text(soup.select_one(".article-source .citation-part"))

        text = f"{title}\n\n{abstract}"

        return {
            "text": text.strip(),
            "title": title,
            "url": url,
            "source_type": "pubmed",
            "pmid": pmid,
            "nct_id": None,
            "authors": authors,
            "journal": journal,
            "publication_date": pub_date,
            "error": None,
        }

    def _parse_clinicaltrials(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        nct_id = re.search(r"(NCT\d{6,8})", url, re.IGNORECASE)
        nct_id = nct_id.group(1).upper() if nct_id else None

        title = self._text(soup.select_one("h1"))
        # Get all study detail sections
        sections = soup.select(".tr-indent2, .ct-body, [data-testid]")
        body = " ".join(s.get_text(" ", strip=True) for s in sections)

        return {
            "text": f"{title}\n\n{body}".strip(),
            "title": title or nct_id or "Clinical Trial",
            "url": url,
            "source_type": "clinical_trial",
            "pmid": None,
            "nct_id": nct_id,
            "authors": "",
            "journal": "ClinicalTrials.gov",
            "publication_date": "",
            "error": None,
        }

    def _parse_biorxiv(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        title = self._text(soup.select_one("h1#page-title"))
        abstract_el = soup.select_one(".abstract")
        abstract = abstract_el.get_text(" ", strip=True) if abstract_el else ""
        authors = self._text(soup.select_one(".contrib-group"))
        date = self._text(soup.select_one(".pub-date"))

        hostname = urlparse(url).hostname or ""
        source_type = "medrxiv" if "medrxiv" in hostname else "biorxiv"

        return {
            "text": f"{title}\n\n{abstract}".strip(),
            "title": title,
            "url": url,
            "source_type": source_type,
            "pmid": None,
            "nct_id": None,
            "authors": authors,
            "journal": source_type,
            "publication_date": date,
            "error": None,
        }

    def _parse_generic(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        # Remove nav/footer/script noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = (
            self._text(soup.select_one("h1"))
            or self._text(soup.select_one("title"))
            or url
        )

        # Prefer article/main content
        content_el = soup.select_one("article, main, .content, #content, .post")
        text = (
            content_el.get_text(" ", strip=True)
            if content_el
            else soup.get_text(" ", strip=True)
        )
        # Collapse whitespace
        text = re.sub(r"\s{3,}", "\n\n", text)

        return {
            "text": text[:8000].strip(),  # cap at 8k chars for generic pages
            "title": title,
            "url": url,
            "source_type": "web",
            "pmid": None,
            "nct_id": None,
            "authors": "",
            "journal": urlparse(url).hostname or "",
            "publication_date": "",
            "error": None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _text(self, el) -> str:
        if el is None:
            return ""
        return el.get_text(" ", strip=True)

    def _error_result(self, url: str, error: str) -> dict:
        return {
            "text": "",
            "title": url,
            "url": url,
            "source_type": "web",
            "pmid": None,
            "nct_id": None,
            "authors": "",
            "journal": "",
            "publication_date": "",
            "error": error,
        }


# ---------------------------------------------------------------------------
# Document Parser (PDF / DOCX)
# ---------------------------------------------------------------------------


class DocumentParser:
    """Parses uploaded PDF or DOCX bytes into plain text."""

    def parse(self, file_bytes: bytes, filename: str) -> dict:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            return self._parse_pdf(file_bytes, filename)
        elif ext in ("docx", "doc"):
            return self._parse_docx(file_bytes, filename)
        else:
            return {
                "text": "",
                "title": filename,
                "error": f"Unsupported file type: .{ext}",
            }

    def _parse_pdf(self, data: bytes, filename: str) -> dict:
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            text = "\n\n".join(text_parts)
            title = filename.replace(".pdf", "").replace("_", " ")
            return {"text": text.strip(), "title": title, "error": None}
        except Exception as e:
            logger.error("PDF parse error: %s", e)
            return {"text": "", "title": filename, "error": str(e)}

    def _parse_docx(self, data: bytes, filename: str) -> dict:
        try:
            import docx

            doc = docx.Document(io.BytesIO(data))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            title = filename.replace(".docx", "").replace("_", " ")
            return {"text": text.strip(), "title": title, "error": None}
        except Exception as e:
            logger.error("DOCX parse error: %s", e)
            return {"text": "", "title": filename, "error": str(e)}


# Singletons
_url_fetcher: Optional[URLFetcher] = None
_doc_parser: Optional[DocumentParser] = None


def get_url_fetcher() -> URLFetcher:
    global _url_fetcher
    if _url_fetcher is None:
        _url_fetcher = URLFetcher()
    return _url_fetcher


def get_document_parser() -> DocumentParser:
    global _doc_parser
    if _doc_parser is None:
        _doc_parser = DocumentParser()
    return _doc_parser
