#!/usr/bin/env python3
"""
MHT-CET College Data Scraper v2.0

Purpose: Resilient web scraper for MHT-CET cutoff data from shiksha.com
Extracts college metadata, branch-wise cutoffs, category-wise ranks, and PDF brochures
Outputs structured JSON with comprehensive error handling and retry logic

Usage:
    python scraper.py

Environment Variables:
    SCRAPER_DELAY_MIN=1.5  # Min delay between requests (seconds)
    SCRAPER_DELAY_MAX=3.0  # Max delay between requests (seconds)
    SCRAPER_MAX_RETRIES=3  # Max retries for failed requests
    SCRAPER_TIMEOUT=30     # Request timeout (seconds)

Output:
    - mht_cet_data.json: Main structured data file
    - mht_cet_data.parquet: Optional efficient format (if pyarrow available)
    - scraper.log: Detailed logging output
"""

import json
import logging
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Optional PDF processing
try:
    import pdfplumber
    PDF_PROCESSING = True
except ImportError:
    PDF_PROCESSING = False
    logging.warning("pdfplumber not available - PDF extraction disabled")

# Optional parquet output
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


@dataclass
class CollegeRecord:
    """Structured college data record"""
    college: str
    city: str
    state: str
    branch: str
    category: str  # Open, OBC, SC, ST
    closing_rank: Optional[int]
    fees: Optional[str]
    placement_rating: Optional[float]
    naac_rating: Optional[str]
    source_url: str
    source_type: str  # "html" or "pdf"
    scraped_at: str


class MHTCETScraper:
    """Resilient MHT-CET data scraper with retry logic and PDF support"""
    
    def __init__(self):
        self.base_url = "https://www.shiksha.com"
        self.start_url = "https://www.shiksha.com/engineering/mht-cet-college-predictor"
        
        # Configuration from env or defaults
        self.delay_min = float(os.getenv('SCRAPER_DELAY_MIN', '1.5'))
        self.delay_max = float(os.getenv('SCRAPER_DELAY_MAX', '3.0'))
        self.max_retries = int(os.getenv('SCRAPER_MAX_RETRIES', '3'))
        self.timeout = int(os.getenv('SCRAPER_TIMEOUT', '30'))
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.scraped_urls = set()
        self.all_records = []

    def _polite_delay(self):
        """Implement polite crawling delay"""
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)

    def _fetch_with_retry(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Fetch URL with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                self._polite_delay()
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * random.uniform(1, 3)
                    self.logger.warning(f"Rate limited on {url}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code >= 500:
                    # Server error - retry
                    self.logger.warning(f"Server error {response.status_code} on {url}, attempt {attempt + 1}")
                    continue
                else:
                    self.logger.error(f"HTTP {response.status_code} on {url}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for {url}, attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract numeric value from text (ranks, fees, etc.)"""
        if not text:
            return None
        
        # Remove common prefixes/suffixes and convert
        import re
        numbers = re.findall(r'\d+', str(text).replace(',', ''))
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        return None

    def _extract_float(self, text: str) -> Optional[float]:
        """Extract float value from text (ratings, etc.)"""
        if not text:
            return None
        
        import re
        matches = re.findall(r'\d+\.?\d*', str(text))
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return None

    def _parse_college_page(self, soup: BeautifulSoup, url: str) -> List[CollegeRecord]:
        """Parse individual college page for detailed cutoff data"""
        records = []
        
        try:
            # Primary selectors with fallbacks
            college_name_selectors = [
                'h1.college-name',
                'h1[data-testid="college-name"]',
                '.college-header h1',
                'h1',  # Fallback
            ]
            
            city_selectors = [
                '.college-location .city',
                '[data-testid="college-city"]',
                '.location-info .city',
                '.college-details .city'
            ]
            
            # Extract college basic info
            college_name = None
            for selector in college_name_selectors:
                elem = soup.select_one(selector)
                if elem:
                    college_name = elem.get_text().strip()
                    break
                    
            if not college_name:
                # Fallback: look for text containing common college keywords
                title_elem = soup.find('title')
                if title_elem and any(word in title_elem.text.lower() for word in ['college', 'institute', 'university']):
                    college_name = title_elem.text.split('|')[0].strip()
                else:
                    college_name = "Unknown College"
            
            # Extract city
            city = None
            for selector in city_selectors:
                elem = soup.select_one(selector)
                if elem:
                    city = elem.get_text().strip()
                    break
            
            if not city:
                # Fallback: look in meta tags or breadcrumbs
                city = "Unknown City"
            
            state = "Maharashtra"  # MHT-CET specific
            
            # Extract cutoff table data
            cutoff_tables = soup.find_all('table') or soup.find_all(class_=lambda x: x and 'cutoff' in x.lower())
            
            if not cutoff_tables:
                # Fallback: look for structured data in divs/cards
                cutoff_containers = soup.find_all(class_=lambda x: x and any(term in x.lower() for term in ['rank', 'cutoff', 'admission']))
            
            # Parse cutoff data
            parsed_cutoffs = self._parse_cutoff_data(soup)
            
            if parsed_cutoffs:
                for cutoff_data in parsed_cutoffs:
                    record = CollegeRecord(
                        college=college_name,
                        city=city,
                        state=state,
                        branch=cutoff_data.get('branch', 'Not Specified'),
                        category=cutoff_data.get('category', 'Open'),
                        closing_rank=cutoff_data.get('closing_rank'),
                        fees=cutoff_data.get('fees'),
                        placement_rating=cutoff_data.get('placement_rating'),
                        naac_rating=cutoff_data.get('naac_rating'),
                        source_url=url,
                        source_type="html",
                        scraped_at=datetime.now().isoformat()
                    )
                    records.append(record)
            else:
                # Create minimal record if no detailed cutoffs found
                record = CollegeRecord(
                    college=college_name,
                    city=city,
                    state=state,
                    branch="Not Specified",
                    category="Open",
                    closing_rank=None,
                    fees=None,
                    placement_rating=None,
                    naac_rating=None,
                    source_url=url,
                    source_type="html",
                    scraped_at=datetime.now().isoformat()
                )
                records.append(record)
                
        except Exception as e:
            self.logger.error(f"Error parsing college page {url}: {e}")
            
        return records

    def _parse_cutoff_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured cutoff data from page"""
        cutoffs = []
        
        # Common branch names for pattern matching
        branches = ['Computer Science', 'Information Technology', 'Electronics', 'Mechanical', 'Civil', 'Electrical']
        categories = ['Open', 'OBC', 'SC', 'ST']
        
        # Try different parsing strategies
        strategies = [
            self._parse_table_cutoffs,
            self._parse_card_cutoffs,
            self._parse_list_cutoffs
        ]
        
        for strategy in strategies:
            try:
                result = strategy(soup)
                if result:
                    cutoffs.extend(result)
                    break
            except Exception as e:
                self.logger.debug(f"Strategy failed: {e}")
                continue
                
        return cutoffs

    def _parse_table_cutoffs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse cutoff data from HTML tables"""
        cutoffs = []
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
                
            # Try to identify header row
            header_row = rows[0]
            headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]
            
            # Look for relevant columns
            branch_col = next((i for i, h in enumerate(headers) if any(term in h for term in ['branch', 'course', 'stream'])), None)
            category_col = next((i for i, h in enumerate(headers) if any(term in h for term in ['category', 'quota'])), None)
            rank_col = next((i for i, h in enumerate(headers) if any(term in h for term in ['rank', 'cutoff', 'closing'])), None)
            
            if rank_col is not None:
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) <= max(branch_col or 0, category_col or 0, rank_col):
                        continue
                        
                    cutoff_data = {
                        'branch': cells[branch_col].get_text().strip() if branch_col is not None else 'General',
                        'category': cells[category_col].get_text().strip() if category_col is not None else 'Open',
                        'closing_rank': self._extract_number(cells[rank_col].get_text().strip()) if rank_col is not None else None
                    }
                    
                    if cutoff_data['closing_rank']:
                        cutoffs.append(cutoff_data)
                        
        return cutoffs

    def _parse_card_cutoffs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse cutoff data from card-based layouts"""
        cutoffs = []
        
        # Look for card containers
        cards = soup.find_all(class_=lambda x: x and any(term in x.lower() for term in ['card', 'item', 'row']))
        
        for card in cards:
            text_content = card.get_text().lower()
            if any(term in text_content for term in ['rank', 'cutoff', 'closing']):
                # Extract data from card
                branch_elem = card.find(string=lambda x: x and any(b.lower() in x.lower() for b in ['computer', 'mechanical', 'civil', 'electronics']))
                rank_numbers = [self._extract_number(elem.get_text()) for elem in card.find_all(string=lambda x: x and any(c.isdigit() for c in x))]
                rank_numbers = [r for r in rank_numbers if r and r > 1000]  # Filter reasonable ranks
                
                if rank_numbers:
                    cutoffs.append({
                        'branch': branch_elem if branch_elem else 'General',
                        'category': 'Open',
                        'closing_rank': min(rank_numbers)
                    })
                    
        return cutoffs

    def _parse_list_cutoffs(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse cutoff data from list layouts"""
        cutoffs = []
        
        # Look for list items containing rank information
        list_items = soup.find_all(['li', 'div'], class_=lambda x: x and 'item' in x.lower())
        
        for item in list_items:
            text = item.get_text()
            if any(term in text.lower() for term in ['rank', 'cutoff']):
                rank = self._extract_number(text)
                if rank and rank > 100:  # Reasonable rank range
                    cutoffs.append({
                        'branch': 'General',
                        'category': 'Open',
                        'closing_rank': rank
                    })
                    
        return cutoffs

    def _extract_pdf_data(self, pdf_url: str) -> List[Dict[str, Any]]:
        """Extract cutoff data from PDF brochures"""
        if not PDF_PROCESSING:
            self.logger.warning(f"PDF processing not available for {pdf_url}")
            return []
            
        try:
            response = self._fetch_with_retry(pdf_url)
            if not response:
                return []
                
            # Save temporarily and process
            temp_path = f"temp_pdf_{int(time.time())}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
                
            cutoffs = []
            
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    # Extract table data
                    tables = page.extract_tables()
                    for table in tables or []:
                        cutoffs.extend(self._parse_pdf_table(table))
                    
                    # Extract text patterns
                    text = page.extract_text()
                    if text:
                        cutoffs.extend(self._parse_pdf_text(text))
                        
            # Cleanup
            os.unlink(temp_path)
            
            return cutoffs
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_url}: {e}")
            return []

    def _parse_pdf_table(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        """Parse cutoff data from PDF table"""
        cutoffs = []
        
        if not table or len(table) < 2:
            return cutoffs
            
        # Identify column structure
        header_row = table[0]
        for row in table[1:]:
            if len(row) >= 2:
                # Extract numerical ranks
                for cell in row:
                    if cell:
                        rank = self._extract_number(cell)
                        if rank and 1000 <= rank <= 200000:  # Reasonable MHT-CET rank range
                            cutoffs.append({
                                'branch': 'General',
                                'category': 'Open',
                                'closing_rank': rank
                            })
                            break
                            
        return cutoffs

    def _parse_pdf_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse cutoff data from PDF text content"""
        cutoffs = []
        
        # Look for rank patterns in text
        import re
        rank_patterns = [
            r'(?:closing|cutoff|rank)[\s:]+(\d{4,6})',
            r'(\d{4,6})[\s]*(?:closing|cutoff|rank)',
        ]
        
        for pattern in rank_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                rank = int(match)
                if 1000 <= rank <= 200000:
                    cutoffs.append({
                        'branch': 'General',
                        'category': 'Open',
                        'closing_rank': rank
                    })
                    
        return cutoffs

    def _find_pagination_urls(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Find pagination URLs to continue scraping"""
        urls = []
        
        # Common pagination selectors
        pagination_selectors = [
            'a[href*="page"]',
            '.pagination a',
            '.pager a',
            'a[aria-label="Next"]',
            'a:contains("Next")',
            'a:contains(">")',
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and href not in self.scraped_urls:
                    full_url = urljoin(current_url, href)
                    urls.append(full_url)
                    
        return urls

    def scrape_all_data(self) -> List[CollegeRecord]:
        """Main scraping orchestration"""
        self.logger.info("Starting MHT-CET data scraping...")
        
        urls_to_process = [self.start_url]
        processed_count = 0
        max_pages = 50  # Safety limit
        
        while urls_to_process and processed_count < max_pages:
            url = urls_to_process.pop(0)
            
            if url in self.scraped_urls:
                continue
                
            self.logger.info(f"Processing page {processed_count + 1}: {url}")
            self.scraped_urls.add(url)
            
            response = self._fetch_with_retry(url)
            if not response:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract college data from current page
            records = self._parse_college_page(soup, url)
            self.all_records.extend(records)
            
            # Find more pages
            new_urls = self._find_pagination_urls(soup, url)
            urls_to_process.extend([u for u in new_urls if u not in self.scraped_urls])
            
            # Look for PDF links
            pdf_links = soup.find_all('a', href=lambda x: x and x.lower().endswith('.pdf'))
            for pdf_link in pdf_links:
                pdf_url = urljoin(url, pdf_link.get('href'))
                self.logger.info(f"Processing PDF: {pdf_url}")
                pdf_data = self._extract_pdf_data(pdf_url)
                
                for pdf_cutoff in pdf_data:
                    # Create record from PDF data
                    record = CollegeRecord(
                        college="Unknown College (PDF)",
                        city="Unknown City",
                        state="Maharashtra",
                        branch=pdf_cutoff.get('branch', 'General'),
                        category=pdf_cutoff.get('category', 'Open'),
                        closing_rank=pdf_cutoff.get('closing_rank'),
                        fees=None,
                        placement_rating=None,
                        naac_rating=None,
                        source_url=pdf_url,
                        source_type="pdf",
                        scraped_at=datetime.now().isoformat()
                    )
                    self.all_records.append(record)
            
            processed_count += 1
            
        self.logger.info(f"Scraping completed. Total records: {len(self.all_records)}")
        return self.all_records

    def save_data(self, output_path: str = "mht_cet_data.json"):
        """Save scraped data to JSON and optionally Parquet"""
        # Convert to dictionaries
        data_dicts = [asdict(record) for record in self.all_records]
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data saved to {output_path}")
        
        # Save Parquet if available
        if PARQUET_AVAILABLE:
            try:
                df = pd.DataFrame(data_dicts)
                parquet_path = output_path.replace('.json', '.parquet')
                df.to_parquet(parquet_path, index=False)
                self.logger.info(f"Data also saved to {parquet_path}")
            except Exception as e:
                self.logger.warning(f"Could not save Parquet: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        if not self.all_records:
            return {}
            
        df = pd.DataFrame([asdict(r) for r in self.all_records])
        
        return {
            'total_records': len(self.all_records),
            'unique_colleges': df['college'].nunique(),
            'unique_branches': df['branch'].nunique(),
            'categories': df['category'].value_counts().to_dict(),
            'source_types': df['source_type'].value_counts().to_dict(),
            'records_with_ranks': df['closing_rank'].notna().sum(),
            'rank_range': {
                'min': df['closing_rank'].min(),
                'max': df['closing_rank'].max(),
                'mean': df['closing_rank'].mean()
            } if df['closing_rank'].notna().any() else None
        }


def main():
    """Main execution function"""
    scraper = MHTCETScraper()
    
    try:
        # Run scraping
        records = scraper.scrape_all_data()
        
        # Save results
        scraper.save_data()
        
        # Print statistics
        stats = scraper.get_stats()
        print("\nScraping Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        if scraper.all_records:
            print("Saving partial results...")
            scraper.save_data("mht_cet_data_partial.json")
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    main()
