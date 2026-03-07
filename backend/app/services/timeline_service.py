import re
from datetime import datetime


# NCT numbers starting with these prefixes correspond to approximate registration years
# NCT format: NCT + 8 digits, first 2 after NCT indicate era

current_year = datetime.now().year

NCT_YEAR_MAP = {
    "023": 2015,
    "024": 2015,
    "025": 2016,
    "026": 2016,
    "027": 2017,
    "028": 2017,
    "029": 2018,
    "030": 2018,
    "031": 2019,
    "032": 2019,
    "033": 2020,
    "034": 2020,
    "035": 2021,
    "036": 2021,
    "037": 2022,
    "038": 2022,
    "039": 2023,
    "040": 2023,
    "041": 2024,
    "042": 2024,
    "043": 2025,
    "058": 2023,
    "059": 2024,
}


def _detect_flags(title: str, text: str = "") -> list[str]:
    """
    Detect credibility flags from a document's title and text.
    Returns a list of flag strings, e.g. ["withdrawn", "retracted"].
    Empty list means no issues detected.
    """
    flags = []
    title_lower = title.lower()
    text_lower = text.lower() if text else ""

    # Withdrawn
    if "withdrawn" in title_lower or "withdrawal" in title_lower:
        flags.append("withdrawn")
    elif "withdrawn" in text_lower or "withdrawal statement" in text_lower:
        flags.append("withdrawn")

    # Retracted
    if "retraction" in title_lower or "retracted" in title_lower:
        flags.append("retracted")
    elif (
        "retraction notice" in text_lower
        or "this article has been retracted" in text_lower
    ):
        flags.append("retracted")

    # Correction / Erratum
    if (
        "correction" in title_lower
        or "erratum" in title_lower
        or "corrigendum" in title_lower
    ):
        flags.append("correction")

    # Preprint — not peer reviewed
    if "preprint" in title_lower or "not peer" in text_lower:
        flags.append("preprint")

    return flags


def classify_study_type(source_type, title):
    title_lower = title.lower()

    if source_type == "clinical_trial":
        if "phase iii" in title_lower:
            return "Phase III"
        if "phase ii" in title_lower:
            return "Phase II"
        if "phase i" in title_lower:
            return "Phase I"
        return "Clinical Trial"

    if source_type == "pubmed":
        return "Published Study"

    if source_type == "biorxiv":
        return "Preprint"

    return "Research"


def estimate_year_from_nct(nct_id: str) -> int | None:
    """Estimate registration year from NCT number prefix."""
    match = re.search(r"NCT(\d{8})", nct_id)
    if match:
        prefix = match.group(1)[:3]
        return NCT_YEAR_MAP.get(prefix)
    return None


def estimate_year_from_pmid(pmid: str) -> int | None:
    """Rough year estimate based on PMID ranges (PMIDs are sequential)."""
    match = re.search(r"(\d+)", pmid)
    if not match:
        return None
    pmid_num = int(match.group(1))
    if pmid_num >= 40_000_000:
        return 2025
    if pmid_num >= 38_000_000:
        return 2024
    if pmid_num >= 36_000_000:
        return 2023
    if pmid_num >= 34_000_000:
        return 2022
    if pmid_num >= 32_000_000:
        return 2021
    if pmid_num >= 30_000_000:
        return 2020
    if pmid_num >= 28_000_000:
        return 2019
    if pmid_num >= 26_000_000:
        return 2018
    if pmid_num >= 24_000_000:
        return 2017
    if pmid_num >= 22_000_000:
        return 2016
    return None


def build_research_timeline(citations):
    timeline = []
    seen_source_ids = set()

    for citation in citations:
        title = citation.get("title", "")
        url = citation.get("url", "")
        source_id = citation.get("source_id", "")
        source_type = citation.get("source_type", "")
        publication_date = citation.get("publication_date")
        # text is optional — only present if your citation dicts include it
        text = citation.get("text", "")

        # Deduplicate
        if source_id and source_id in seen_source_ids:
            continue
        if source_id:
            seen_source_ids.add(source_id)

        year = None

        if publication_date:
            date_match = re.search(r"(20\d{2})", str(publication_date))
            if date_match:
                year = int(date_match.group(1))

        if year is None:
            year_match = re.search(r"(20\d{2})", title) or re.search(r"(20\d{2})", url)
            if year_match:
                year = int(year_match.group(1))

        if year is None and source_type == "clinical_trial":
            year = estimate_year_from_nct(source_id)

        if year is None and source_type == "pubmed":
            year = estimate_year_from_pmid(source_id)

        if year is None:
            continue

        # Detect credibility flags
        flags = _detect_flags(title, text)

        # year should not of future
        if year > current_year:
            continue

        timeline.append(
            {
                "year": year,
                "title": title[:120],
                "url": url,
                "type": classify_study_type(source_type, title),
                "source_type": source_type,
                "flags": flags,  # e.g. ["withdrawn"], ["retracted"], []
            }
        )

    timeline.sort(key=lambda x: (x["year"] is None, x["year"] or 0))
    return timeline
