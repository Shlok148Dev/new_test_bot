# scraper.py
"""
Purpose:
    Resilient scraper to harvest MHT-CET cutoff information from Shiksha's college predictor pages.
    Produces `mht_cet_data.json` and (optionally) `mht_cet_data.parquet`.

How to run:
    1. Create and populate a .env file (see .env.example later in the project).
    2. Install dependencies: `pip install -r requirements.txt`
    3. Run: `python scraper.py`
    4. Output files will be written to ./data/mht_cet_data.json and ./data/mht_cet_data.parquet (if pandas/pyarrow installed).

Environment variables (via .env):
    - USER_AGENT (optional): user agent string to use for requests. A default is provided.
    - REQUEST_MIN_DELAY (optional): minimum polite delay between requests in seconds (default 1.5).
    - REQUEST_MAX_DELAY (optional): maximum polite delay between requests in seconds (default 3.0).
    - MAX_PAGES (optional): optional limit to pages for testing; default None (crawl until pagination ends).
    - OUTPUT_DIR (optional): output directory (default ./data).

Notes / behavior:
    - Respects polite delays and exponential backoff on 429/5xx responses.
    - Uses robust parsing heuristics (CSS selectors + text matching + DOM-relative traversal).
    - If a PDF cutoff brochure is found, downloads and attempts table/text extraction via pdfplumber.
    - Logs progress, warnings, and parse misses to stdout and `scraper.log`.
    - Schema per record: {
          "college": str,
          "city": Optional[str],
          "state": Optional[str],
          "branch": str,
          "category": str,          # e.g., Open/OBC/SC/ST
          "closing_rank": Optional[int],
          "fees": Optional[str],
          "placement_rating": Optional[float],
          "naac_rating": Optional[str],
          "source_url": str,
          "source_type": "html"|"pdf",
          "retrieved_at": ISO8601 timestamp
      }
"""

import os
import re
import time
import json
import math
import logging
import random
import pathlib
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Optional PDF parsing; handle import errors gracefully
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Optional parquet output
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# ---- Configuration (overridable by environment) ----
BASE_URL = "https://www.shiksha.com/engineering/mht-cet-college-predictor"
USER_AGENT = os.getenv("USER_AGENT",
                       "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36 CET-Mentor-Scraper/1.0")
REQUEST_MIN_DELAY = float(os.getenv("REQUEST_MIN_DELAY", "1.5"))
REQUEST_MAX_DELAY = float(os.getenv("REQUEST_MAX_DELAY", "3.0"))
MAX_PAGES = os.getenv("MAX_PAGES")  # optional: for testing
OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "./data"))
LOG_FILE = "scraper.log"

# Create output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("mht_cet_scraper")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# Session with default headers
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

# Rate limiting helpers
def polite_sleep():
    delay = random.uniform(REQUEST_MIN_DELAY, REQUEST_MAX_DELAY)
    logger.debug(f"Sleeping for {delay:.2f}s to be polite.")
    time.sleep(delay)

def exponential_backoff(attempt: int, base: float = 1.0, cap: float = 32.0):
    sleep = min(cap, base * (2 ** attempt) + random.random())
    logger.warning(f"Backoff sleep for {sleep:.1f}s (attempt {attempt}).")
    time.sleep(sleep)

# Utility: safe request with retries
def safe_get(url: str, max_retries: int = 5, timeout: int = 20) -> Optional[requests.Response]:
    attempt = 0
    while attempt <= max_retries:
        try:
            logger.info(f"GET {url} (attempt {attempt + 1})")
            resp = SESSION.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Server returned {resp.status_code} for {url}")
                exponential_backoff(attempt)
                attempt += 1
                continue
            logger.error(f"Unexpected status {resp.status_code} for {url}")
            return resp
        except requests.RequestException as e:
            logger.exception(f"Request exception for {url}: {e}")
            exponential_backoff(attempt)
            attempt += 1
    logger.error(f"Exceeded max retries for {url}")
    return None

# ---- Parsing helpers ----

def text_or_none(el) -> Optional[str]:
    if not el:
        return None
    txt = el.get_text(separator=" ", strip=True)
    return txt if txt else None

def parse_int_maybe(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Remove commas, non-digits
    m = re.search(r"(\d{1,7})", s.replace(",", ""))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def parse_float_maybe(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def normalize_category(cat: str) -> str:
    # canonical categories
    cat = (cat or "").strip().lower()
    if not cat:
        return "Open"
    if "open" in cat:
        return "Open"
    if "obc" in cat:
        return "OBC"
    if "sc" in cat and "st" not in cat:
        return "SC"
    if "st" in cat:
        return "ST"
    return cat.title()

# Extract PDF tables/text and try to find (college, branch, category, closing rank)
def extract_from_pdf_bytes(pdf_bytes: bytes, source_url: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not pdfplumber:
        logger.warning("pdfplumber not available; skipping PDF parsing.")
        return results
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_all = []
            for p in pdf.pages:
                try:
                    # Try table extraction
                    tables = p.extract_tables()
                    if tables:
                        for table in tables:
                            # Flatten table rows and attempt to detect common patterns
                            for row in table:
                                row_text = " | ".join([cell or "" for cell in row])
                                text_all.append(row_text)
                    # Also append raw text for regex parsing
                    t = p.extract_text()
                    if t:
                        text_all.append(t)
                except Exception as e:
                    logger.debug(f"pdf page parse error: {e}")
            combined = "\n".join(text_all)
            # Heuristic: find lines mentioning branch & category & rank
            lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
            for ln in lines:
                # Simple heuristics: "Computer Engineering ... Open 23567"
                m = re.search(r"(?P<branch>[A-Za-z &/-]{3,60}).{0,20}(?P<cat>Open|OBC|SC|ST|General)?.{0,20}(?P<rank>\d{2,7})", ln, re.I)
                if m:
                    rec = {
                        "college": None,
                        "city": None,
                        "state": None,
                        "branch": m.group("branch").strip(),
                        "category": normalize_category(m.group("cat") or "Open"),
                        "closing_rank": parse_int_maybe(m.group("rank")),
                        "fees": None,
                        "placement_rating": None,
                        "naac_rating": None,
                        "source_url": source_url,
                        "source_type": "pdf",
                        "retrieved_at": datetime.utcnow().isoformat() + "Z"
                    }
                    results.append(rec)
            logger.info(f"Extracted {len(results)} candidate rows from PDF {source_url}")
    except Exception as e:
        logger.exception(f"PDF parsing failed for {source_url}: {e}")
    return results

# ---- HTML parsing for a page ----

def find_pagination_next(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    # Try common patterns: rel="next", anchor text Next, aria-label next, or page links
    next_link = soup.find("link", {"rel": "next"})
    if next_link and next_link.get("href"):
        return urljoin(base_url, next_link["href"])
    # anchor with rel next
    a = soup.find("a", attrs={"rel": "next"})
    if a and a.get("href"):
        return urljoin(base_url, a["href"])
    # Look for 'Next' text buttons
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True).lower() in ("next", ">", ">>"):
            return urljoin(base_url, a["href"])
    # last resort: find pagination and pick next based on 'active' class
    pag = soup.select_one(".pagination, .pagnation, ul.pagination")
    if pag:
        active = pag.select_one(".active")
        if active:
            nxt = active.find_next_sibling("li")
            if nxt and nxt.find("a") and nxt.find("a").get("href"):
                return urljoin(base_url, nxt.find("a")["href"])
    return None

def parse_college_cards(soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
    """
    Parse college/row cards from the predictor list page.
    This function uses heuristics, because site structure may change.
    """
    records: List[Dict[str, Any]] = []
    # Typical blocks: article cards, results list, table rows
    candidates = []
    # Candidate selectors to try
    selectors = [
        "div.college-card, div.predictor-card, div.result-card, div.college-list-item",
        "article, .srp-card, .listing-card",
        "div[class*='college'], div[class*='card']"
    ]
    for sel in selectors:
        found = soup.select(sel)
        if found:
            candidates = found
            logger.debug(f"Found {len(found)} candidates with selector '{sel}'")
            break
    if not candidates:
        # fallback: search for rows with college names (heuristic)
        for h in soup.find_all(re.compile("^h[1-6]$")):
            if h.get_text(strip=True) and len(h.get_text(strip=True)) < 120:
                parent = h.find_parent()
                if parent:
                    candidates.append(parent)
    logger.info(f"Parsing {len(candidates)} candidate blocks from {page_url}")

    for block in candidates:
        try:
            # Attempt multiple strategies to pull data
            college_name = None
            # priority: anchor with college name
            a = block.find("a", href=True, text=True)
            if a and len(a.get_text(strip=True)) > 3:
                college_name = a.get_text(strip=True)
            if not college_name:
                # headings inside block
                h = block.find(["h2", "h3", "h4"])
                if h:
                    college_name = h.get_text(strip=True)
            # try meta info
            city = None
            state = None
            # Often location is in a span or small tag
            loc = block.find(lambda tag: tag.name in ("span", "p", "small") and "location" in (tag.get("class") or []) )
            if not loc:
                # text that contains comma separated location like "Mumbai, Maharashtra"
                text_nodes = block.find_all(text=True)
                for t in text_nodes:
                    if "," in t and len(t) < 60 and any(k in t.lower() for k in ("mumbai","pune","nagpur","nashik","thane","aurangabad","maharashtra")):
                        parts = [p.strip() for p in t.split(",") if p.strip()]
                        if parts:
                            city = parts[0]
                            if len(parts) >= 2:
                                state = parts[1]
                                break
            # Branches/department and cutoff info
            # Look for small tables or lists inside block
            branches = []
            # Try to find table rows in block
            for table in block.find_all("table"):
                for tr in table.find_all("tr"):
                    cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th","td"])]
                    if not cells:
                        continue
                    # Heuristics: if row contains branch and rank
                    br = None
                    cat = None
                    rank = None
                    # Some tables: Branch | Category | Closing Rank
                    if len(cells) >= 2:
                        # search for a cell that looks like rank
                        for c in cells:
                            if re.search(r"\b\d{2,7}\b", c):
                                rank = parse_int_maybe(c)
                        # branch candidate is first cell
                        br = cells[0]
                        # detect category if present
                        for c in cells:
                            if re.search(r"\b(Open|OBC|SC|ST|General)\b", c, re.I):
                                cat = normalize_category(c)
                        branches.append((br, cat, rank))
            # Try lists (ul/li) with text like "Computer Engg - Open - Closing Rank 23456"
            if not branches:
                for li in block.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if re.search(r"\d{2,7}", txt):
                        br = None
                        cat = None
                        rank = None
                        # branch before '-' or '('
                        m_br = re.match(r"(.{3,80}?)\s*(?:-|,|\(|:)", txt)
                        if m_br:
                            br = m_br.group(1).strip()
                        if re.search(r"\b(Open|OBC|SC|ST|General)\b", txt, re.I):
                            cat = normalize_category(re.search(r"\b(Open|OBC|SC|ST|General)\b", txt, re.I).group(1))
                        rank = parse_int_maybe(txt)
                        branches.append((br, cat, rank))
            # If still empty, attempt to use text blocks with keywords "Closing rank" or "Closing Rank"
            if not branches:
                bigtxt = block.get_text(" ", strip=True)
                lines = [ln.strip() for ln in re.split(r"\.\s+|\n", bigtxt) if ln.strip()]
                for ln in lines:
                    if "closing" in ln.lower() and re.search(r"\d{2,7}", ln):
                        br = None
                        cat = None
                        rank = parse_int_maybe(ln)
                        # try capture branch before 'closing'
                        m = re.match(r"(.{3,60}?)\s+closing", ln, re.I)
                        if m:
                            br = m.group(1).strip()
                        branches.append((br, cat, rank))
            # Another fallback: if block links to a college details page, follow it and parse table there
            detail_url = None
            link = block.find("a", href=True)
            if link:
                detail_url = urljoin(page_url, link["href"])
            # If we have branch tuples, create records
            if branches:
                for br, cat, rank in branches:
                    rec = {
                        "college": college_name or (link.get("title") if link else None) or "Unknown",
                        "city": city,
                        "state": state,
                        "branch": (br or "Unknown").strip() if br else "Unknown",
                        "category": normalize_category(cat or "Open"),
                        "closing_rank": rank,
                        "fees": None,
                        "placement_rating": None,
                        "naac_rating": None,
                        "source_url": page_url if not detail_url else detail_url,
                        "source_type": "html",
                        "retrieved_at": datetime.utcnow().isoformat() + "Z"
                    }
                    records.append(rec)
            else:
                # As last resort, if college_name exists, push a minimal record (so we don't lose row)
                if college_name:
                    rec = {
                        "college": college_name,
                        "city": city,
                        "state": state,
                        "branch": "All",
                        "category": "Open",
                        "closing_rank": None,
                        "fees": None,
                        "placement_rating": None,
                        "naac_rating": None,
                        "source_url": page_url,
                        "source_type": "html",
                        "retrieved_at": datetime.utcnow().isoformat() + "Z"
                    }
                    records.append(rec)
        except Exception as e:
            logger.exception(f"Error parsing a candidate block: {e}")
    # Deduplicate (college+branch+category)
    unique = {}
    for r in records:
        key = (r.get("college"), r.get("branch"), r.get("category"))
        if key not in unique:
            unique[key] = r
        else:
            # prefer non-null closing_rank
            if unique[key].get("closing_rank") is None and r.get("closing_rank") is not None:
                unique[key] = r
    deduped = list(unique.values())
    logger.info(f"Parsed {len(deduped)} deduped records from page {page_url}")
    return deduped

# ---- Main crawl loop ----

def crawl_predictor(start_url: str = BASE_URL, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    logger.info(f"Starting crawl at {start_url}")
    url = start_url
    page_count = 0
    all_records: List[Dict[str, Any]] = []
    visited_pages = set()

    while url:
        if max_pages:
            try:
                if page_count >= int(max_pages):
                    logger.info(f"Reached MAX_PAGES={max_pages}. Stopping crawl.")
                    break
            except Exception:
                pass

        if url in visited_pages:
            logger.info(f"Encountered already visited page {url}. Stopping pagination loop.")
            break
        visited_pages.add(url)

        resp = safe_get(url)
        if not resp:
            logger.error(f"Failed to fetch {url}. Stopping.")
            break

        polite_sleep()

        soup = BeautifulSoup(resp.text, "html.parser")
        # parse page-level PDFs (some pages may link to a brochure)
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_links.append(urljoin(url, href))
        # parse HTML records from page
        page_records = parse_college_cards(soup, url)
        all_records.extend(page_records)

        # Try to parse PDFs linked directly on the page
        for pdf_url in set(pdf_links):
            try:
                logger.info(f"Downloading PDF brochure: {pdf_url}")
                pdf_resp = safe_get(pdf_url)
                if pdf_resp and pdf_resp.content:
                    # attempt to extract table rows
                    pdf_recs = []
                    if pdfplumber:
                        import io
                        pdf_recs = extract_from_pdf_bytes(pdf_resp.content, pdf_url)
                    else:
                        logger.debug("pdfplumber not installed; skipping pdf contents extraction.")
                        # still add a record noting pdf existence
                        pdf_recs = [{
                            "college": None,
                            "city": None,
                            "state": None,
                            "branch": "Unknown",
                            "category": "Open",
                            "closing_rank": None,
                            "fees": None,
                            "placement_rating": None,
                            "naac_rating": None,
                            "source_url": pdf_url,
                            "source_type": "pdf",
                            "retrieved_at": datetime.utcnow().isoformat() + "Z"
                        }]
                    all_records.extend(pdf_recs)
                polite_sleep()
            except Exception as e:
                logger.exception(f"Error processing PDF {pdf_url}: {e}")

        # find next page via pagination heuristics
        next_page = find_pagination_next(soup, url)
        page_count += 1
        if next_page:
            parsed = urlparse(next_page)
            if not parsed.scheme:
                next_page = urljoin(url, next_page)
            url = next_page
            logger.info(f"Next page -> {url}")
        else:
            logger.info("No next page found. Finishing crawl.")
            break

    # Postprocess: normalize fields and remove blatantly invalid entries
    normalized = []
    seen_keys = set()
    for r in all_records:
        college = (r.get("college") or "").strip() if r.get("college") else None
        branch = (r.get("branch") or "All").strip()
        category = normalize_category(r.get("category") or "Open")
        key = (college, branch, category)
        if not college:
            logger.debug("Skipping record with missing college name.")
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized.append({
            "college": college,
            "city": r.get("city"),
            "state": r.get("state"),
            "branch": branch,
            "category": category,
            "closing_rank": r.get("closing_rank"),
            "fees": r.get("fees"),
            "placement_rating": r.get("placement_rating"),
            "naac_rating": r.get("naac_rating"),
            "source_url": r.get("source_url"),
            "source_type": r.get("source_type") or "html",
            "retrieved_at": r.get("retrieved_at") or datetime.utcnow().isoformat() + "Z"
        })
    logger.info(f"Crawl finished. Total normalized records: {len(normalized)}")
    return normalized

# ---- Save outputs ----

def save_json(records: List[Dict[str, Any]], path: pathlib.Path):
    logger.info(f"Writing {len(records)} records to JSON: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def save_parquet(records: List[Dict[str, Any]], path: pathlib.Path):
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available; skipping parquet output.")
        return
    try:
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
        logger.info(f"Wrote parquet to {path}")
    except Exception as e:
        logger.exception(f"Failed to write parquet: {e}")

# ---- Command-line entrypoint ----

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MHT-CET cutoff scraper (Shiksha predictor)")
    parser.add_argument("--start-url", default=BASE_URL, help="Start URL (default shiksha predictor)")
    parser.add_argument("--max-pages", default=MAX_PAGES, help="Optional max pages to crawl (int)")
    parser.add_argument("--out-json", default=str(OUTPUT_DIR / "mht_cet_data.json"), help="Output JSON path")
    parser.add_argument("--out-parquet", default=str(OUTPUT_DIR / "mht_cet_data.parquet"), help="Output parquet path")
    parser.add_argument("--no-parquet", action="store_true", help="Do not try to write parquet even if pandas present")
    args = parser.parse_args()

    mp = None
    if args.max_pages:
        try:
            mp = int(args.max_pages)
        except Exception:
            mp = None

    records = crawl_predictor(start_url=args.start_url, max_pages=mp)
    save_json(records, pathlib.Path(args.out_json))
    if not args.no_parquet:
        save_parquet(records, pathlib.Path(args.out_parquet))
    logger.info("Done.")

if __name__ == "__main__":
    main()
