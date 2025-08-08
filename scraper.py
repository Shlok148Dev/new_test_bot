#!/usr/bin/env python3
"""
MHT-CET College Data Scraper for CET-Mentor v2.0
Scrapes Shiksha.com MHT-CET College Predictor and extracts cutoff data
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
import time
import re
import pandas as pd
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional
import pdfplumber
from io import BytesIO
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MHTCETScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.base_url = "https://www.shiksha.com"
        self.predictor_url = "https://www.shiksha.com/mht-cet/college-predictor"
        self.data = []
        self.max_retries = 3
        self.delay = 2

    def make_request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None

    def extract_college_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract college profile links from predictor page"""
        college_links = []
        
        # Look for college links in various possible selectors
        selectors = [
            'a[href*="/college/"]',
            '.college-card a',
            '.college-list a',
            'a[href*="mht-cet"]'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and 'college' in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in college_links:
                        college_links.append(full_url)
        
        logger.info(f"Found {len(college_links)} college links")
        return college_links

    def extract_college_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract college metadata from college page"""
        metadata = {
            'college': '',
            'city': '',
            'naac_rating': '',
            'fees': '',
            'placement_rating': ''
        }
        
        try:
            # College name
            name_selectors = ['h1', '.college-name', '.page-title', 'title']
            for selector in name_selectors:
                element = soup.select_one(selector)
                if element:
                    metadata['college'] = element.get_text().strip()
                    break
            
            # City
            city_selectors = [
                '[class*="location"]',
                '[class*="city"]',
                '.address',
                '[data-label*="location"]'
            ]
            for selector in city_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    # Extract city from address
                    if ',' in text:
                        metadata['city'] = text.split(',')[0].strip()
                    else:
                        metadata['city'] = text
                    break
            
            # NAAC Rating
            naac_selectors = [
                '[class*="naac"]',
                '[class*="rating"]',
                '[data-label*="naac"]'
            ]
            for selector in naac_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    if any(grade in text.upper() for grade in ['A++', 'A+', 'A', 'B+', 'B']):
                        metadata['naac_rating'] = text
                        break
            
            # Fees
            fees_selectors = [
                '[class*="fee"]',
                '[class*="cost"]',
                '[data-label*="fee"]'
            ]
            for selector in fees_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    if '₹' in text or 'Rs' in text or any(char.isdigit() for char in text):
                        metadata['fees'] = text
                        break
            
            # Placement rating (look for percentage or rating)
            placement_selectors = [
                '[class*="placement"]',
                '[class*="package"]',
                '[data-label*="placement"]'
            ]
            for selector in placement_selectors:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text().strip()
                    if '%' in text or any(word in text.lower() for word in ['lpa', 'package', 'salary']):
                        metadata['placement_rating'] = text
                        break
                        
        except Exception as e:
            logger.warning(f"Error extracting metadata from {url}: {e}")
            
        return metadata

    def extract_cutoff_data(self, soup: BeautifulSoup, college_metadata: Dict) -> List[Dict]:
        """Extract cutoff data from college page"""
        cutoffs = []
        
        try:
            # Look for cutoff tables
            table_selectors = [
                'table',
                '.cutoff-table',
                '.admission-table',
                '[class*="cutoff"]'
            ]
            
            for selector in table_selectors:
                tables = soup.select(selector)
                for table in tables:
                    cutoffs.extend(self.parse_cutoff_table(table, college_metadata))
            
            # Look for cutoff data in other formats (divs, lists)
            if not cutoffs:
                cutoff_containers = soup.select('[class*="cutoff"], [class*="rank"], [class*="admission"]')
                for container in cutoff_containers:
                    cutoffs.extend(self.parse_cutoff_container(container, college_metadata))
                    
        except Exception as e:
            logger.warning(f"Error extracting cutoff data: {e}")
            
        return cutoffs

    def parse_cutoff_table(self, table, college_metadata: Dict) -> List[Dict]:
        """Parse cutoff data from HTML table"""
        cutoffs = []
        
        try:
            rows = table.select('tr')
            if len(rows) < 2:
                return cutoffs
                
            # Get header row
            header_row = rows[0]
            headers = [th.get_text().strip().lower() for th in header_row.select('th, td')]
            
            # Find relevant column indices
            branch_col = self.find_column_index(headers, ['branch', 'course', 'program', 'discipline'])
            category_col = self.find_column_index(headers, ['category', 'quota', 'reservation'])
            rank_col = self.find_column_index(headers, ['rank', 'cutoff', 'closing', 'last'])
            
            if branch_col == -1 or rank_col == -1:
                return cutoffs
                
            # Parse data rows
            for row in rows[1:]:
                cells = row.select('td')
                if len(cells) < max(branch_col, rank_col) + 1:
                    continue
                    
                branch = cells[branch_col].get_text().strip() if branch_col != -1 else ''
                category = cells[category_col].get_text().strip() if category_col != -1 else 'Open'
                rank_text = cells[rank_col].get_text().strip() if rank_col != -1 else ''
                
                # Extract numeric rank
                rank = self.extract_numeric_rank(rank_text)
                
                if branch and rank:
                    cutoff_data = {
                        'college': college_metadata.get('college', ''),
                        'branch': branch,
                        'category': self.normalize_category(category),
                        'closing_rank': rank,
                        'fees': college_metadata.get('fees', ''),
                        'city': college_metadata.get('city', ''),
                        'naac_rating': college_metadata.get('naac_rating', '')
                    }
                    cutoffs.append(cutoff_data)
                    
        except Exception as e:
            logger.warning(f"Error parsing cutoff table: {e}")
            
        return cutoffs

    def parse_cutoff_container(self, container, college_metadata: Dict) -> List[Dict]:
        """Parse cutoff data from non-table containers"""
        cutoffs = []
        
        try:
            text = container.get_text()
            # Look for patterns like "CSE - Open: 1234, OBC: 2345"
            
            branch_patterns = [
                r'(?:Computer Science|CSE|IT|Information Technology|Mechanical|MECH|Electrical|EEE|Civil|Chemical|Electronics|ECE)',
                r'[A-Z]{2,4}(?:\s+Engineering)?'
            ]
            
            for branch_pattern in branch_patterns:
                matches = re.finditer(branch_pattern, text, re.IGNORECASE)
                for match in matches:
                    branch = match.group()
                    # Look for rank data after branch name
                    remaining_text = text[match.end():match.end()+200]
                    
                    # Extract category-wise ranks
                    category_patterns = [
                        (r'Open[:\s-]+(\d+)', 'Open'),
                        (r'OBC[:\s-]+(\d+)', 'OBC'),
                        (r'SC[:\s-]+(\d+)', 'SC'),
                        (r'ST[:\s-]+(\d+)', 'ST'),
                        (r'General[:\s-]+(\d+)', 'Open')
                    ]
                    
                    for pattern, category in category_patterns:
                        rank_match = re.search(pattern, remaining_text, re.IGNORECASE)
                        if rank_match:
                            rank = int(rank_match.group(1))
                            cutoff_data = {
                                'college': college_metadata.get('college', ''),
                                'branch': branch,
                                'category': category,
                                'closing_rank': rank,
                                'fees': college_metadata.get('fees', ''),
                                'city': college_metadata.get('city', ''),
                                'naac_rating': college_metadata.get('naac_rating', '')
                            }
                            cutoffs.append(cutoff_data)
                            
        except Exception as e:
            logger.warning(f"Error parsing cutoff container: {e}")
            
        return cutoffs

    def find_column_index(self, headers: List[str], keywords: List[str]) -> int:
        """Find column index based on keywords"""
        for i, header in enumerate(headers):
            for keyword in keywords:
                if keyword in header:
                    return i
        return -1

    def extract_numeric_rank(self, rank_text: str) -> Optional[int]:
        """Extract numeric rank from text"""
        # Remove common prefixes/suffixes and extract number
        numbers = re.findall(r'\d+', rank_text.replace(',', ''))
        if numbers:
            return int(numbers[0])
        return None

    def normalize_category(self, category: str) -> str:
        """Normalize category names"""
        category = category.upper().strip()
        
        category_map = {
            'OPEN': 'Open',
            'GENERAL': 'Open',
            'GEN': 'Open',
            'OBC': 'OBC',
            'OBCNCL': 'OBC',
            'SC': 'SC',
            'ST': 'ST',
            'EWS': 'EWS'
        }
        
        for key, value in category_map.items():
            if key in category:
                return value
                
        return 'Open'  # Default to Open category

    def find_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find PDF links that might contain cutoff data"""
        pdf_links = []
        
        # Look for PDF links
        pdf_selectors = [
            'a[href$=".pdf"]',
            'a[href*=".pdf"]',
            'a[href*="cutoff"]',
            'a[href*="admission"]'
        ]
        
        for selector in pdf_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    pdf_links.append(full_url)
                    
        return pdf_links

    def extract_pdf_data(self, pdf_url: str) -> List[Dict]:
        """Extract cutoff data from PDF"""
        cutoffs = []
        
        try:
            response = self.make_request(pdf_url)
            if not response:
                return cutoffs
                
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Look for tabular data
                        tables = page.extract_tables()
                        for table in tables:
                            cutoffs.extend(self.parse_pdf_table(table))
                            
                        # Also parse text for patterns
                        cutoffs.extend(self.parse_pdf_text(text))
                        
        except Exception as e:
            logger.warning(f"Error processing PDF {pdf_url}: {e}")
            
        return cutoffs

    def parse_pdf_table(self, table: List[List]) -> List[Dict]:
        """Parse cutoff data from PDF table"""
        cutoffs = []
        
        if not table or len(table) < 2:
            return cutoffs
            
        try:
            headers = [str(cell).lower().strip() if cell else '' for cell in table[0]]
            
            branch_col = self.find_column_index(headers, ['branch', 'course', 'program'])
            category_col = self.find_column_index(headers, ['category', 'quota'])
            rank_col = self.find_column_index(headers, ['rank', 'cutoff', 'closing'])
            
            for row in table[1:]:
                if len(row) < max(branch_col, rank_col) + 1:
                    continue
                    
                branch = str(row[branch_col]).strip() if branch_col != -1 and row[branch_col] else ''
                category = str(row[category_col]).strip() if category_col != -1 and row[category_col] else 'Open'
                rank_text = str(row[rank_col]).strip() if rank_col != -1 and row[rank_col] else ''
                
                rank = self.extract_numeric_rank(rank_text)
                
                if branch and rank:
                    cutoff_data = {
                        'college': '',  # Will be filled later
                        'branch': branch,
                        'category': self.normalize_category(category),
                        'closing_rank': rank,
                        'fees': '',
                        'city': '',
                        'naac_rating': ''
                    }
                    cutoffs.append(cutoff_data)
                    
        except Exception as e:
            logger.warning(f"Error parsing PDF table: {e}")
            
        return cutoffs

    def parse_pdf_text(self, text: str) -> List[Dict]:
        """Parse cutoff data from PDF text"""
        cutoffs = []
        
        try:
            # Similar to parse_cutoff_container but for PDF text
            lines = text.split('\n')
            
            for line in lines:
                # Look for patterns indicating cutoff data
                if any(keyword in line.lower() for keyword in ['cutoff', 'closing', 'rank', 'admission']):
                    # Extract branch and rank information
                    # This would need more sophisticated parsing based on actual PDF formats
                    pass
                    
        except Exception as e:
            logger.warning(f"Error parsing PDF text: {e}")
            
        return cutoffs

    def scrape_colleges(self):
        """Main scraping method"""
        logger.info("Starting MHT-CET college data scraping...")
        
        # Start with predictor page
        response = self.make_request(self.predictor_url)
        if not response:
            logger.error("Failed to access predictor page")
            return
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract college links
        college_links = self.extract_college_links(soup)
        
        if not college_links:
            logger.warning("No college links found, trying alternative approach...")
            # Try to find colleges through search or other methods
            college_links = self.find_alternative_college_links()
        
        logger.info(f"Processing {len(college_links)} colleges...")
        
        for i, college_url in enumerate(college_links, 1):
            logger.info(f"Processing college {i}/{len(college_links)}: {college_url}")
            
            try:
                # Get college page
                response = self.make_request(college_url)
                if not response:
                    continue
                    
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract metadata
                metadata = self.extract_college_metadata(soup, college_url)
                
                # Extract cutoff data
                cutoffs = self.extract_cutoff_data(soup, metadata)
                
                # Look for PDFs
                pdf_links = self.find_pdf_links(soup, college_url)
                for pdf_url in pdf_links:
                    pdf_cutoffs = self.extract_pdf_data(pdf_url)
                    # Update PDF cutoffs with college metadata
                    for cutoff in pdf_cutoffs:
                        cutoff.update({
                            'college': metadata.get('college', ''),
                            'fees': metadata.get('fees', ''),
                            'city': metadata.get('city', ''),
                            'naac_rating': metadata.get('naac_rating', '')
                        })
                    cutoffs.extend(pdf_cutoffs)
                
                self.data.extend(cutoffs)
                logger.info(f"Extracted {len(cutoffs)} cutoff records from {metadata.get('college', 'Unknown')}")
                
                # Respectful delay
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error processing {college_url}: {e}")
                continue

    def find_alternative_college_links(self) -> List[str]:
        """Alternative method to find college links"""
        college_links = []
        
        # Try searching for popular engineering colleges in Maharashtra
        popular_colleges = [
            "VJTI Mumbai", "COEP Pune", "ICT Mumbai", "SPPU", "Government College of Engineering"
        ]
        
        for college in popular_colleges:
            search_url = f"{self.base_url}/search"
            params = {'q': f"{college} MHT-CET"}
            
            response = self.make_request(search_url, params)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.select('a[href*="/college/"]')
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in college_links:
                            college_links.append(full_url)
                            
                time.sleep(1)  # Respectful delay
                
        return college_links

    def save_data(self, filename: str = 'mht_cet_data.json'):
        """Save scraped data to JSON file"""
        try:
            # Remove duplicates
            seen = set()
            unique_data = []
            
            for item in self.data:
                # Create a unique identifier
                identifier = f"{item['college']}_{item['branch']}_{item['category']}"
                if identifier not in seen:
                    seen.add(identifier)
                    unique_data.append(item)
            
            logger.info(f"Saving {len(unique_data)} unique records to {filename}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Data successfully saved to {filename}")
            
            # Also save as CSV for easier analysis
            if unique_data:
                df = pd.DataFrame(unique_data)
                csv_filename = filename.replace('.json', '.csv')
                df.to_csv(csv_filename, index=False)
                logger.info(f"Data also saved as CSV: {csv_filename}")
                
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def run(self):
        """Run the complete scraping process"""
        try:
            self.scrape_colleges()
            self.save_data()
            
            logger.info(f"Scraping completed. Total records: {len(self.data)}")
            
            # Print summary
            if self.data:
                colleges = set(item['college'] for item in self.data)
                branches = set(item['branch'] for item in self.data)
                categories = set(item['category'] for item in self.data)
                
                logger.info(f"Summary:")
                logger.info(f"- Colleges: {len(colleges)}")
                logger.info(f"- Branches: {len(branches)}")
                logger.info(f"- Categories: {len(categories)}")
                
        except Exception as e:
            logger.error(f"Error in main scraping process: {e}")
        finally:
            self.session.close()

def main():
    """Main function"""
    scraper = MHTCETScraper()
    scraper.run()

if __name__ == "__main__":
    main()
