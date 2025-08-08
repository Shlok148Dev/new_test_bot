#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CET-Mentor v2.0 Data Scraper

Scrapes MHT-CET college predictor data from Shiksha and outputs a clean JSON file
that the application can use for RAG lookups.

Key features:
- Resilient HTML parsing with multiple selector fallbacks
- Pagination detection via rel="next", link text, and next-like classes
- Structured, normalized, and deduplicated output
- Logging, retries with backoff, and configurable rate limits
- CLI options to control output and verbosity
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter, Retry

# --- Configuration ---
BASE_URL = "https://www.shiksha.com/engineering/mht-cet-college-predictor"
OUTPUT_JSON = "mht_cet_data.json"
OUTPUT_CSV = "mht_cet_data.csv"
REQUEST_TIMEOUT_SECS = 20
REQUEST_DELAY_SECS = 1.5
MAX_PAGES = 200  # hard ceiling to avoid infinite loops if site changes

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("scraper")

BRANCH_DEFAULT = "Computer Science and Engineering"


@dataclass
class CollegeRecord:
    college: str
    branch: str
    closing_rank: Optional[int]
    source_url: str


class ShikshaScraper:
    def __init__(self, base_url: str = BASE_URL, delay: float = REQUEST_DELAY_SECS):
        self.base_url = base_url
        self.delay = delay
        self.session = self._init_session()

    def _init_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(HEADERS)
        return session

    def fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            logger.info(f"Fetching: {url}")
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT_SECS)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None

    def parse_cards(self, soup: BeautifulSoup, page_url: str) -> List[CollegeRecord]:
        """
        Attempt to extract college cards with robust fallbacks.
        """
        records: List[CollegeRecord] = []

        # Primary guess: cards contain data-csm-parent-name="predictor-Result-card"
        card_candidates: List[Tag] = soup.find_all(
            lambda tag: isinstance(tag, Tag)
            and tag.name in ("div", "section", "li")
            and (
                tag.get("data-csm-parent-name") == "predictor-Result-card"
                or "predictor" in " ".join(tag.get("class", []))
                or "tuple" in " ".join(tag.get("class", [])).lower()
                or "card" in " ".join(tag.get("class", [])).lower()
            )
        )

        # Fallback: look for any block that mentions Closing Rank text
        if not card_candidates:
            card_candidates = soup.find_all(
                lambda tag: isinstance(tag, Tag)
                and tag.name in ("div", "li", "section")
                and tag.find(string=re.compile(r"Closing\s+(All\s+India\s+)?rank", re.I))
            )

        logger.info(f"Found {len(card_candidates)} card candidates")

        for card in card_candidates:
            try:
                college_name = self._extract_college_name(card)
                closing_rank = self._extract_closing_rank(card)
                branch = self._extract_branch(card) or BRANCH_DEFAULT

                if college_name:
                    records.append(
                        CollegeRecord(
                            college=self._clean_college_name(college_name),
                            branch=branch,
                            closing_rank=closing_rank,
                            source_url=page_url,
                        )
                    )
            except Exception as e:
                logger.debug(f"Card parse error: {e}")

        return records

    def _extract_college_name(self, card: Tag) -> Optional[str]:
        # Try common patterns
        # 1) Bold or strong titles
        title = card.find(["h3", "h2", "h4", "p"], class_=re.compile(r"(title|name|bold)", re.I))
        if title and title.get_text(strip=True):
            return title.get_text(strip=True)

        # 2) First anchor text that looks like a college name (and links to /college/)
        for a in card.find_all("a", href=True):
            text = a.get_text(strip=True)
            if not text:
                continue
            href = a["href"].lower()
            if "/college" in href or "/university" in href or "shiksha.com" in href:
                return text

        # 3) Any prominent text
        strong = card.find(["strong", "b"])
        if strong and strong.get_text(strip=True):
            return strong.get_text(strip=True)

        # 4) First heading text
        heading = card.find(["h1", "h2", "h3", "h4"]) or card.find("p")
        if heading and heading.get_text(strip=True):
            return heading.get_text(strip=True)

        return None

    def _extract_closing_rank(self, card: Tag) -> Optional[int]:
        # Look for phrases like 'Closing All India rank' or 'Closing Rank'
        text = card.get_text(" ", strip=True)
        match = re.search(r"Closing\s+(All\s+India\s+)?rank\s*:?\s*([\d,]+)", text, flags=re.I)
        if match:
            return self._to_int(match.group(2))

        # Look for any number near 'Closing' and 'rank'
        if re.search(r"Closing", text, re.I) and re.search(r"rank", text, re.I):
            nums = re.findall(r"\b[\d,]{2,}\b", text)
            if nums:
                return self._to_int(nums[0])

        # Structured: label-value pairs in adjacent tags
        label = card.find(lambda t: isinstance(t, Tag) and t.name in ("p", "div", "span") and re.search(r"Closing.*rank", t.get_text(" ", strip=True), re.I))
        if label:
            sibling = label.find_next_sibling(["p", "div", "span"]) or label
            nums = re.findall(r"\b[\d,]{2,}\b", sibling.get_text(" ", strip=True))
            if nums:
                return self._to_int(nums[0])

        return None

    def _extract_branch(self, card: Tag) -> Optional[str]:
        text = card.get_text(" ", strip=True)
        # Simple heuristics for branch detection
        branch_keywords = [
            "Computer Science", "CSE", "Information Technology", "Mechanical",
            "Civil", "Electrical", "Electronics", "AI", "Data Science",
        ]
        for kw in branch_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
                return kw
        return None

    def _clean_college_name(self, name: str) -> str:
        name = re.sub(r"\s+\|.*$", "", name).strip()
        name = re.sub(r"\s*\([^)]+\)$", "", name).strip()  # drop trailing (…)
        name = re.sub(r"\s{2,}", " ", name)
        return name

    def _to_int(self, s: str) -> Optional[int]:
        try:
            return int(re.sub(r"[^0-9]", "", s))
        except Exception:
            return None

    def find_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        # Try rel="next"
        link = soup.find("a", rel=re.compile(r"\bnext\b", re.I))
        if link and link.get("href"):
            return urljoin(current_url, link["href"])

        # Try anchor with "next" text
        link = soup.find("a", string=re.compile(r"\bnext\b", re.I))
        if link and link.get("href"):
            return urljoin(current_url, link["href"])

        # Try class names
        link = soup.find("a", class_=re.compile(r"next|pagination__next|pager__next|next-page", re.I))
        if link and link.get("href"):
            return urljoin(current_url, link["href"])

        return None

    def scrape(self, start_url: Optional[str] = None, max_pages: int = MAX_PAGES) -> List[CollegeRecord]:
        url = start_url or self.base_url
        page = 1
        aggregated: List[CollegeRecord] = []

        while url and page <= max_pages:
            soup = self.fetch_soup(url)
            if not soup:
                break

            page_records = self.parse_cards(soup, url)
            logger.info(f"Page {page}: extracted {len(page_records)} records")
            aggregated.extend(page_records)

            next_url = self.find_next_page(soup, url)
            if not next_url:
                logger.info("No additional pages found.")
                break

            url = next_url
            page += 1
            time.sleep(self.delay)

        return aggregated


def to_dataframe(records: Iterable[CollegeRecord]) -> pd.DataFrame:
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["college", "branch", "closing_rank", "source_url"]) 

    # Normalize
    df["college"] = (
        df["college"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    df["branch"] = df["branch"].fillna(BRANCH_DEFAULT)

    # Convert closing_rank to numeric with safe coercion
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce").astype("Int64")

    # Drop rows with no college name
    df = df[df["college"].notna() & (df["college"].str.len() > 1)]

    # Deduplicate by college + branch keeping the best (lowest) rank if available
    df.sort_values(by=["college", "branch", "closing_rank"], ascending=[True, True, True], inplace=True)
    df = df.drop_duplicates(subset=["college", "branch"], keep="first")

    # Sort by closing rank ascending (best first), nulls last
    df["closing_rank_filled"] = df["closing_rank"].fillna(10**9)
    df.sort_values(by=["closing_rank_filled", "college"], inplace=True)
    df.drop(columns=["closing_rank_filled"], inplace=True)

    return df.reset_index(drop=True)


def save_outputs(df: pd.DataFrame, json_path: str = OUTPUT_JSON, csv_path: str = OUTPUT_CSV) -> None:
    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    # CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} records to {json_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Shiksha MHT-CET predictor data")
    parser.add_argument("--start-url", default=BASE_URL, help="Starting URL for scraping")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Max pages to crawl")
    parser.add_argument("--json", default=OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--csv", default=OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECS, help="Delay between requests (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    scraper = ShikshaScraper(base_url=args.start_url, delay=args.delay)
    records = scraper.scrape(start_url=args.start_url, max_pages=args.max_pages)

    if not records:
        logger.error("No data scraped. The site structure may have changed.")
        sys.exit(2)

    df = to_dataframe(records)
    if df.empty:
        logger.error("No usable rows after cleaning.")
        sys.exit(3)

    save_outputs(df, json_path=args.json, csv_path=args.csv)


if __name__ == "__main__":
    main()