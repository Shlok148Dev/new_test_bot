#!/usr/bin/env python3
"""
MHT-CET College Data Scraper for RAG System

Purpose: Automatically fetch cutoff PDFs from DTE Maharashtra website and supplement
with metadata from educational sites to create structured JSON for vector storage.

Features:
- Auto-download latest + 4 previous years of cutoff PDFs
- Parse PDFs to extract branches and category-wise cutoffs
- Scrape college metadata (fees, placement ratings, type, location)
- Generate structured JSON ready for vector ingestion
- Async I/O for faster scraping
- Robust error handling and retry logic

Usage:
    python scraper.py

Output:
    ./pdfs/           - Downloaded PDF files
    ./data/structured_data.json - Final structured JSON

Dependencies:
    pip install aiohttp beautifulsoup4 pypdf pdfplumber requests lxml pandas numpy
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import hashlib

# Core libraries
import aiohttp
import requests
from bs4 import BeautifulSoup

# PDF processing
import pypdf
import pdfplumber

# Data processing
import pandas as pd
import numpy as np


class MHTCETScraper:
    """Production-grade scraper for MHT-CET college admission data"""
    
    def __init__(self, base_dir: str = ".", years_back: int = 5):
        """
        Initialize scraper with configuration
        
        Args:
            base_dir: Base directory for output files
            years_back: Number of years back to fetch (including current year)
        """
        self.base_dir = Path(base_dir)
        self.years_back = years_back
        self.current_year = datetime.now().year
        
        # Create directories
        self.pdfs_dir = self.base_dir / "pdfs"
        self.data_dir = self.base_dir / "data"
        self.pdfs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # URLs and patterns
        self.dte_base_url = "https://dtemaharashtra.gov.in"
        self.dte_cutoff_url = "https://dtemaharashtra.gov.in/cutoff"
        
        # College metadata sources
        self.metadata_sources = [
            "https://collegedunia.com",
            "https://shiksha.com",
            "https://careers360.com"
        ]
        
        # Category normalization mapping
        self.category_mapping = {
            'open': 'OPEN', 'general': 'OPEN', 'gen': 'OPEN',
            'obc': 'OBC', 'other backward class': 'OBC', 'obcl': 'OBC',
            'sc': 'SC', 'scheduled caste': 'SC',
            'st': 'ST', 'scheduled tribe': 'ST',
            'ews': 'EWS', 'economically weaker section': 'EWS',
            'vjnt': 'VJNT', 'vj': 'VJNT', 'nt': 'NT',
            'sbc': 'SBC', 'special backward class': 'SBC'
        }
        
        # Branch name normalization
        self.branch_mapping = {
            'computer engineering': 'Computer Engineering',
            'computer science': 'Computer Science and Engineering',
            'information technology': 'Information Technology',
            'electronics engineering': 'Electronics Engineering',
            'electronics & telecommunication': 'Electronics and Telecommunication',
            'mechanical engineering': 'Mechanical Engineering',
            'civil engineering': 'Civil Engineering',
            'electrical engineering': 'Electrical Engineering',
            'chemical engineering': 'Chemical Engineering',
            'automobile engineering': 'Automobile Engineering',
            'production engineering': 'Production Engineering',
            'instrumentation engineering': 'Instrumentation Engineering',
            'textile engineering': 'Textile Engineering',
            'biotechnology': 'Biotechnology',
            'aerospace engineering': 'Aerospace Engineering'
        }
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    async def download_pdfs(self) -> List[str]:
        """
        Download cutoff PDFs for specified years
        
        Returns:
            List of downloaded PDF file paths
        """
        self.logger.info(f"Starting PDF download for {self.years_back} years")
        
        # Get available PDF URLs
        pdf_urls = await self._discover_pdf_urls()
        
        if not pdf_urls:
            self.logger.warning("No PDF URLs found, using fallback method")
            pdf_urls = self._get_fallback_pdf_urls()
        
        # Download PDFs concurrently
        downloaded_files = []
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=5)
        ) as session:
            tasks = []
            for year, url in pdf_urls.items():
                if year >= self.current_year - self.years_back + 1:
                    task = self._download_single_pdf(session, year, url)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, str):  # Successful download
                    downloaded_files.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Download failed: {result}")
        
        self.logger.info(f"Downloaded {len(downloaded_files)} PDF files")
        return downloaded_files

    async def _discover_pdf_urls(self) -> Dict[int, str]:
        """
        Discover available PDF URLs from DTE website
        
        Returns:
            Dictionary mapping year to PDF URL
        """
        pdf_urls = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # First, try to get the cutoff page
                async with session.get(self.dte_cutoff_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Look for PDF links
                        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))
                        
                        for link in pdf_links:
                            href = link.get('href', '')
                            text = link.get_text(strip=True).lower()
                            
                            # Extract year from link text or URL
                            year_match = re.search(r'20(\d{2})', text + href)
                            if year_match:
                                year = int(f"20{year_match.group(1)}")
                                if 'cutoff' in text or 'merit' in text:
                                    full_url = urljoin(self.dte_base_url, href)
                                    pdf_urls[year] = full_url
                
                # Try alternative pages
                alternative_urls = [
                    f"{self.dte_base_url}/admissions",
                    f"{self.dte_base_url}/results",
                    f"{self.dte_base_url}/downloads"
                ]
                
                for url in alternative_urls:
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                                pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))
                                for link in pdf_links:
                                    href = link.get('href', '')
                                    text = link.get_text(strip=True).lower()
                                    
                                    year_match = re.search(r'20(\d{2})', text + href)
                                    if year_match and ('cutoff' in text or 'merit' in text):
                                        year = int(f"20{year_match.group(1)}")
                                        full_url = urljoin(self.dte_base_url, href)
                                        pdf_urls[year] = full_url
                    except Exception as e:
                        self.logger.debug(f"Failed to check {url}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error discovering PDF URLs: {e}")
        
        return pdf_urls

    def _get_fallback_pdf_urls(self) -> Dict[int, str]:
        """
        Fallback method to construct likely PDF URLs
        
        Returns:
            Dictionary mapping year to constructed PDF URL
        """
        pdf_urls = {}
        
        # Common URL patterns for DTE Maharashtra
        url_patterns = [
            f"{self.dte_base_url}/cutoff/{{year}}_cutoff.pdf",
            f"{self.dte_base_url}/downloads/cutoff{{year}}.pdf",
            f"{self.dte_base_url}/results/merit_list_{{year}}.pdf"
        ]
        
        for year in range(self.current_year, self.current_year - self.years_back, -1):
            for pattern in url_patterns:
                url = pattern.format(year=year)
                pdf_urls[year] = url
                break  # Use first pattern for each year
        
        return pdf_urls

    async def _download_single_pdf(self, session: aiohttp.ClientSession, 
                                 year: int, url: str) -> str:
        """
        Download a single PDF file with retry logic
        
        Args:
            session: aiohttp session
            year: Year of the PDF
            url: URL to download from
            
        Returns:
            Path to downloaded file
        """
        filename = f"{year}_cutoff.pdf"
        file_path = self.pdfs_dir / filename
        
        # Skip if already exists
        if file_path.exists():
            self.logger.info(f"PDF {filename} already exists, skipping")
            return str(file_path)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {filename} (attempt {attempt + 1}/{max_retries})")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Verify it's a PDF
                        if content.startswith(b'%PDF'):
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            
                            self.logger.info(f"Successfully downloaded {filename}")
                            return str(file_path)
                        else:
                            self.logger.warning(f"Downloaded content for {year} is not a valid PDF")
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                self.logger.error(f"Download attempt {attempt + 1} failed for {year}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to download PDF for year {year} after {max_retries} attempts")

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF to extract college, branch, and cutoff data
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with parsed data
        """
        self.logger.info(f"Parsing PDF: {file_path}")
        
        try:
            # Try pdfplumber first (better for tables)
            data = self._parse_with_pdfplumber(file_path)
            if data:
                return data
        except Exception as e:
            self.logger.warning(f"pdfplumber failed: {e}")
        
        try:
            # Fallback to pypdf
            data = self._parse_with_pypdf(file_path)
            if data:
                return data
        except Exception as e:
            self.logger.warning(f"pypdf failed: {e}")
        
        # Last resort: create minimal structure
        year = self._extract_year_from_filename(file_path)
        return {
            'year': year,
            'colleges': [],
            'source_pdf': os.path.basename(file_path)
        }

    def _parse_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF using pdfplumber (better for tables)"""
        colleges_data = []
        year = self._extract_year_from_filename(file_path)
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract tables
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        
                        # Convert to DataFrame for easier processing
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = df.dropna(how='all').fillna('')
                        
                        # Parse table data
                        parsed_colleges = self._parse_table_data(df, year)
                        colleges_data.extend(parsed_colleges)
                        
                except Exception as e:
                    self.logger.debug(f"Error parsing page {page_num}: {e}")
                    continue
        
        return {
            'year': year,
            'colleges': colleges_data,
            'source_pdf': os.path.basename(file_path)
        }

    def _parse_with_pypdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF using pypdf (fallback method)"""
        colleges_data = []
        year = self._extract_year_from_filename(file_path)
        
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Parse text data
                    parsed_colleges = self._parse_text_data(text, year)
                    colleges_data.extend(parsed_colleges)
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing page {page_num}: {e}")
                    continue
        
        return {
            'year': year,
            'colleges': colleges_data,
            'source_pdf': os.path.basename(file_path)
        }

    def _parse_table_data(self, df: pd.DataFrame, year: int) -> List[Dict[str, Any]]:
        """Parse tabular data from DataFrame"""
        colleges = []
        
        # Common column patterns
        college_col_patterns = ['college', 'institute', 'institution', 'name']
        branch_col_patterns = ['branch', 'course', 'program', 'stream']
        category_col_patterns = ['open', 'obc', 'sc', 'st', 'ews', 'general']
        
        # Find relevant columns
        columns = [col.lower().strip() for col in df.columns if col]
        
        college_col = None
        branch_col = None
        category_cols = {}
        
        for col in columns:
            if any(pattern in col for pattern in college_col_patterns):
                college_col = col
            elif any(pattern in col for pattern in branch_col_patterns):
                branch_col = col
            else:
                for pattern in category_col_patterns:
                    if pattern in col:
                        normalized_category = self._normalize_category(pattern)
                        category_cols[normalized_category] = col
        
        if not college_col:
            return colleges
        
        # Process rows
        for _, row in df.iterrows():
            try:
                college_name = str(row.get(college_col, '')).strip()
                if not college_name or len(college_name) < 3:
                    continue
                
                branch_name = str(row.get(branch_col, 'General')).strip()
                branch_name = self._normalize_branch(branch_name)
                
                # Extract category cutoffs
                category_cutoffs = {}
                for category, col in category_cols.items():
                    cutoff_value = str(row.get(col, '')).strip()
                    if cutoff_value and cutoff_value.isdigit():
                        category_cutoffs[category] = int(cutoff_value)
                
                if category_cutoffs:  # Only add if we have cutoff data
                    college_data = {
                        'college_name': college_name,
                        'year': year,
                        'branches': [{
                            'branch_name': branch_name,
                            'category_cutoffs': category_cutoffs
                        }]
                    }
                    colleges.append(college_data)
                    
            except Exception as e:
                self.logger.debug(f"Error processing row: {e}")
                continue
        
        return colleges

    def _parse_text_data(self, text: str, year: int) -> List[Dict[str, Any]]:
        """Parse text data using regex patterns"""
        colleges = []
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_college = None
        current_branch = None
        
        for line in lines:
            # Try to identify college names (usually in caps or title case)
            if re.match(r'^[A-Z][A-Z\s,\.]+$', line) and len(line) > 10:
                current_college = line.strip()
                continue
            
            # Try to identify branch names
            for branch_pattern in self.branch_mapping.keys():
                if branch_pattern.lower() in line.lower():
                    current_branch = self._normalize_branch(line)
                    break
            
            # Look for cutoff data (numbers with category indicators)
            cutoff_match = re.findall(r'(\w+)[\s:]+(\d+)', line)
            if cutoff_match and current_college:
                category_cutoffs = {}
                for category, cutoff in cutoff_match:
                    normalized_category = self._normalize_category(category)
                    if normalized_category:
                        category_cutoffs[normalized_category] = int(cutoff)
                
                if category_cutoffs:
                    college_data = {
                        'college_name': current_college,
                        'year': year,
                        'branches': [{
                            'branch_name': current_branch or 'General',
                            'category_cutoffs': category_cutoffs
                        }]
                    }
                    colleges.append(college_data)
        
        return colleges

    def _extract_year_from_filename(self, file_path: str) -> int:
        """Extract year from PDF filename"""
        filename = os.path.basename(file_path)
        year_match = re.search(r'20(\d{2})', filename)
        if year_match:
            return int(f"20{year_match.group(1)}")
        return self.current_year

    def _normalize_category(self, category: str) -> Optional[str]:
        """Normalize category name to standard format"""
        category = category.lower().strip()
        return self.category_mapping.get(category)

    def _normalize_branch(self, branch: str) -> str:
        """Normalize branch name to standard format"""
        branch = branch.lower().strip()
        for pattern, normalized in self.branch_mapping.items():
            if pattern in branch:
                return normalized
        return branch.title()

    async def scrape_college_metadata(self, college_name: str) -> Dict[str, Any]:
        """
        Scrape additional metadata for a college
        
        Args:
            college_name: Name of the college
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'college_type': 'Unknown',
            'location': 'Unknown',
            'fees': 'Not Available',
            'placement_rating': 'Not Available',
            'naac_rating': 'Not Available'
        }
        
        # Try multiple sources
        for source_url in self.metadata_sources:
            try:
                source_metadata = await self._scrape_from_source(college_name, source_url)
                # Merge non-empty values
                for key, value in source_metadata.items():
                    if value and value != 'Unknown' and value != 'Not Available':
                        metadata[key] = value
                        
                # If we have good data, no need to check other sources
                if all(v != 'Unknown' and v != 'Not Available' for v in metadata.values()):
                    break
                    
            except Exception as e:
                self.logger.debug(f"Failed to scrape from {source_url}: {e}")
                continue
        
        return metadata

    async def _scrape_from_source(self, college_name: str, source_url: str) -> Dict[str, Any]:
        """Scrape metadata from a specific source"""
        metadata = {}
        
        async with aiohttp.ClientSession() as session:
            # Search for college
            search_url = f"{source_url}/search"
            search_params = {'query': college_name}
            
            try:
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find college links
                        college_links = soup.find_all('a', href=True)
                        
                        for link in college_links:
                            if college_name.lower() in link.get_text().lower():
                                college_url = urljoin(source_url, link['href'])
                                metadata = await self._extract_metadata_from_page(session, college_url)
                                break
                                
            except Exception as e:
                self.logger.debug(f"Search failed for {college_name} on {source_url}: {e}")
        
        return metadata

    async def _extract_metadata_from_page(self, session: aiohttp.ClientSession, 
                                        url: str) -> Dict[str, Any]:
        """Extract metadata from college page"""
        metadata = {}
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract college type
                    if 'government' in html.lower() or 'govt' in html.lower():
                        metadata['college_type'] = 'Government'
                    elif 'private' in html.lower():
                        metadata['college_type'] = 'Private'
                    
                    # Extract location
                    location_patterns = [
                        r'(?:location|address|city)[\s:]*([a-zA-Z\s,]+)',
                        r'(?:mumbai|pune|nashik|nagpur|aurangabad)',
                    ]
                    
                    for pattern in location_patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            metadata['location'] = match.group(1).strip().title()
                            break
                    
                    # Extract fees (look for numeric values with currency indicators)
                    fees_patterns = [
                        r'(?:fees|fee|tuition)[\s:]*(?:rs\.?|₹)?\s*([0-9,]+)',
                        r'₹\s*([0-9,]+)',
                        r'rs\.?\s*([0-9,]+)'
                    ]
                    
                    for pattern in fees_patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            fees_value = match.group(1).replace(',', '')
                            if fees_value.isdigit():
                                metadata['fees'] = f"₹{int(fees_value):,}"
                                break
                    
                    # Extract placement rating
                    rating_patterns = [
                        r'(?:placement|rating|score)[\s:]*([0-9.]+)',
                        r'([0-9.]+)[\s/]*(?:out of|/)\s*([0-9.]+)'
                    ]
                    
                    for pattern in rating_patterns:
                        match = re.search(pattern, html, re.IGNORECASE)
                        if match:
                            if len(match.groups()) == 2:
                                rating = f"{match.group(1)}/{match.group(2)}"
                            else:
                                rating = match.group(1)
                            metadata['placement_rating'] = rating
                            break
                    
        except Exception as e:
            self.logger.debug(f"Failed to extract metadata from {url}: {e}")
        
        return metadata

    def _estimate_missing_metadata(self, college_name: str, 
                                 category_cutoffs: Dict[str, int]) -> Dict[str, Any]:
        """Estimate missing metadata based on college name and cutoff patterns"""
        metadata = {}
        
        college_lower = college_name.lower()
        
        # Estimate college type
        if any(keyword in college_lower for keyword in ['iit', 'nit', 'government', 'govt']):
            metadata['college_type'] = 'Government'
        else:
            metadata['college_type'] = 'Private'
        
        # Estimate location based on common patterns
        cities = ['mumbai', 'pune', 'nashik', 'nagpur', 'aurangabad', 'kolhapur', 'solapur']
        for city in cities:
            if city in college_lower:
                metadata['location'] = city.title()
                break
        else:
            metadata['location'] = 'Maharashtra'
        
        # Estimate fees based on college type and cutoff
        if metadata['college_type'] == 'Government':
            metadata['fees'] = '₹50,000 - ₹1,00,000'
        else:
            # Higher cutoff (lower rank) usually means higher fees
            open_cutoff = category_cutoffs.get('OPEN', 100000)
            if open_cutoff < 10000:
                metadata['fees'] = '₹3,00,000 - ₹5,00,000'
            elif open_cutoff < 50000:
                metadata['fees'] = '₹1,50,000 - ₹3,00,000'
            else:
                metadata['fees'] = '₹1,00,000 - ₹2,00,000'
        
        # Estimate placement rating based on cutoff
        open_cutoff = category_cutoffs.get('OPEN', 100000)
        if open_cutoff < 5000:
            metadata['placement_rating'] = '4.0/5.0'
        elif open_cutoff < 20000:
            metadata['placement_rating'] = '3.5/5.0'
        elif open_cutoff < 50000:
            metadata['placement_rating'] = '3.0/5.0'
        else:
            metadata['placement_rating'] = '2.5/5.0'
        
        return metadata

    async def build_json_from_pdfs(self) -> Dict[str, Any]:
        """
        Build comprehensive JSON structure from all PDFs
        
        Returns:
            Complete structured data dictionary
        """
        self.logger.info("Building structured JSON from PDFs")
        
        # Download PDFs first
        pdf_files = await self.download_pdfs()
        
        if not pdf_files:
            self.logger.error("No PDF files available to process")
            return {'colleges': [], 'metadata': {'total_colleges': 0, 'years_processed': []}}
        
        all_colleges_data = {}  # college_name -> data
        years_processed = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            try:
                pdf_data = self.parse_pdf(pdf_file)
                year = pdf_data['year']
                years_processed.append(year)
                
                self.logger.info(f"Processing {len(pdf_data['colleges'])} colleges from {year}")
                
                # Merge colleges data
                for college_data in pdf_data['colleges']:
                    college_name = college_data['college_name']
                    
                    if college_name not in all_colleges_data:
                        all_colleges_data[college_name] = {
                            'college_name': college_name,
                            'years_data': {},
                            'branches': {},
                            'metadata': {}
                        }
                    
                    # Add year-specific data
                    all_colleges_data[college_name]['years_data'][year] = {
                        'branches': college_data['branches'],
                        'source_pdf': pdf_data['source_pdf']
                    }
                    
                    # Collect all branches
                    for branch_data in college_data['branches']:
                        branch_name = branch_data['branch_name']
                        if branch_name not in all_colleges_data[college_name]['branches']:
                            all_colleges_data[college_name]['branches'][branch_name] = {}
                        
                        all_colleges_data[college_name]['branches'][branch_name][year] = branch_data['category_cutoffs']
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        # Enrich with metadata
        self.logger.info("Enriching data with college metadata")
        enriched_colleges = []
        
        total_colleges = len(all_colleges_data)
        for idx, (college_name, college_data) in enumerate(all_colleges_data.items(), 1):
            if idx % 10 == 0:
                self.logger.info(f"Processed metadata for {idx}/{total_colleges} colleges")
            
            try:
                # Get latest year's cutoff data for estimation
                latest_year = max(college_data['years_data'].keys())
                latest_branches = college_data['years_data'][latest_year]['branches']
                
                # Get representative cutoff for estimation
                representative_cutoffs = {}
                if latest_branches:
                    representative_cutoffs = latest_branches[0]['category_cutoffs']
                
                # Try to scrape metadata
                try:
                    metadata = await self.scrape_college_metadata(college_name)
                except Exception as e:
                    self.logger.debug(f"Metadata scraping failed for {college_name}: {e}")
                    metadata = {}
                
                # Fill missing metadata with estimates
                estimated_metadata = self._estimate_missing_metadata(college_name, representative_cutoffs)
                for key, value in estimated_metadata.items():
                    if key not in metadata or metadata[key] in ['Unknown', 'Not Available']:
                        metadata[key] = value
                
                # Build final college structure
                college_entry = {
                    'college_name': college_name,
                    'college_type': metadata.get('college_type', 'Unknown'),
                    'location': metadata.get('location', 'Unknown'),
                    'naac_rating': metadata.get('naac_rating', 'Not Available'),
                    'years_data': college_data['years_data'],
                    'branches_summary': [],
                    'metadata': {
                        'fees': metadata.get('fees', 'Not Available'),
                        'placement_rating': metadata.get('placement_rating', 'Not Available'),
                        'years_available': list(college_data['years_data'].keys()),
                        'total_branches': len(college_data['branches']),
                        'last_updated': datetime.now().isoformat()
                    }
                }
                
                # Create branches summary with multi-year data
                for branch_name, year_cutoffs in college_data['branches'].items():
                    branch_summary = {
                        'branch_name': branch_name,
                        'yearly_cutoffs': year_cutoffs,
                        'latest_cutoffs': {},
                        'trend_analysis': self._analyze_cutoff_trends(year_cutoffs)
                    }
                    
                    # Get latest year's cutoffs for this branch
                    if year_cutoffs:
                        latest_cutoff_year = max(year_cutoffs.keys())
                        branch_summary['latest_cutoffs'] = year_cutoffs[latest_cutoff_year]
                    
                    college_entry['branches_summary'].append(branch_summary)
                
                enriched_colleges.append(college_entry)
                
            except Exception as e:
                self.logger.error(f"Error enriching data for {college_name}: {e}")
                continue
        
        # Build final structure
        structured_data = {
            'colleges': enriched_colleges,
            'metadata': {
                'total_colleges': len(enriched_colleges),
                'years_processed': sorted(set(years_processed)),
                'generation_timestamp': datetime.now().isoformat(),
                'scraper_version': '1.0.0',
                'data_sources': {
                    'primary': 'DTE Maharashtra',
                    'metadata_sources': self.metadata_sources
                },
                'statistics': self._generate_statistics(enriched_colleges)
            }
        }
        
        self.logger.info(f"Built structured data for {len(enriched_colleges)} colleges")
        return structured_data

    def _analyze_cutoff_trends(self, year_cutoffs: Dict[int, Dict[str, int]]) -> Dict[str, Any]:
        """Analyze cutoff trends across years"""
        if len(year_cutoffs) < 2:
            return {'trend': 'insufficient_data'}
        
        trends = {}
        years = sorted(year_cutoffs.keys())
        
        for category in ['OPEN', 'OBC', 'SC', 'ST']:
            category_data = []
            for year in years:
                if category in year_cutoffs[year]:
                    category_data.append(year_cutoffs[year][category])
            
            if len(category_data) >= 2:
                # Calculate trend (negative means ranks are improving/getting lower)
                trend_slope = (category_data[-1] - category_data[0]) / len(category_data)
                
                if trend_slope < -500:
                    trend = 'improving_significantly'
                elif trend_slope < -100:
                    trend = 'improving'
                elif trend_slope > 500:
                    trend = 'declining_significantly'
                elif trend_slope > 100:
                    trend = 'declining'
                else:
                    trend = 'stable'
                
                trends[category] = {
                    'trend': trend,
                    'slope': trend_slope,
                    'latest_rank': category_data[-1],
                    'change_from_first': category_data[-1] - category_data[0]
                }
        
        return trends

    def _generate_statistics(self, colleges_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        stats = {
            'total_colleges': len(colleges_data),
            'college_types': {'Government': 0, 'Private': 0, 'Unknown': 0},
            'locations': {},
            'branches': {},
            'cutoff_ranges': {}
        }
        
        all_cutoffs = []
        
        for college in colleges_data:
            # College type distribution
            college_type = college.get('college_type', 'Unknown')
            stats['college_types'][college_type] = stats['college_types'].get(college_type, 0) + 1
            
            # Location distribution
            location = college.get('location', 'Unknown')
            stats['locations'][location] = stats['locations'].get(location, 0) + 1
            
            # Branch analysis
            for branch in college.get('branches_summary', []):
                branch_name = branch['branch_name']
                stats['branches'][branch_name] = stats['branches'].get(branch_name, 0) + 1
                
                # Collect cutoff data
                latest_cutoffs = branch.get('latest_cutoffs', {})
                for category, cutoff in latest_cutoffs.items():
                    if isinstance(cutoff, int):
                        all_cutoffs.append(cutoff)
        
        # Cutoff statistics
        if all_cutoffs:
            all_cutoffs.sort()
            stats['cutoff_ranges'] = {
                'min': min(all_cutoffs),
                'max': max(all_cutoffs),
                'median': all_cutoffs[len(all_cutoffs) // 2],
                'percentile_25': all_cutoffs[len(all_cutoffs) // 4],
                'percentile_75': all_cutoffs[3 * len(all_cutoffs) // 4]
            }
        
        return stats

    def save_json(self, data: Dict[str, Any], 
                  output_path: str = "./data/structured_data.json") -> str:
        """
        Save structured data to JSON file
        
        Args:
            data: Structured data dictionary
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Structured data saved to {output_path}")
            
            # Also save a compact version for faster loading
            compact_path = output_path.parent / f"{output_path.stem}_compact.json"
            with open(compact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
            
            # Save summary for quick overview
            summary = {
                'total_colleges': data['metadata']['total_colleges'],
                'years_processed': data['metadata']['years_processed'],
                'generation_timestamp': data['metadata']['generation_timestamp'],
                'statistics': data['metadata']['statistics']
            }
            
            summary_path = output_path.parent / f"{output_path.stem}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving JSON to {output_path}: {e}")
            raise

    async def run_full_pipeline(self) -> str:
        """
        Run the complete scraping and processing pipeline
        
        Returns:
            Path to final JSON file
        """
        self.logger.info("Starting MHT-CET data scraping pipeline")
        start_time = time.time()
        
        try:
            # Build structured data from PDFs
            structured_data = await self.build_json_from_pdfs()
            
            # Save to JSON
            output_path = self.save_json(structured_data)
            
            # Log completion
            duration = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            self.logger.info(f"Processed {structured_data['metadata']['total_colleges']} colleges")
            self.logger.info(f"Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def create_vector_store_compatible_format(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert structured data to format compatible with vector_store.py
        
        Args:
            structured_data: Output from build_json_from_pdfs
            
        Returns:
            List of records compatible with vector store
        """
        records = []
        
        for college in structured_data['colleges']:
            college_name = college['college_name']
            college_type = college['college_type']
            location = college['location']
            
            for branch in college['branches_summary']:
                branch_name = branch['branch_name']
                latest_cutoffs = branch.get('latest_cutoffs', {})
                
                # Create a record for each category that has cutoff data
                for category, cutoff_rank in latest_cutoffs.items():
                    record = {
                        'college': college_name,
                        'branch': branch_name,
                        'category': category,
                        'closing_rank': cutoff_rank,
                        'college_type': college_type,
                        'city': location.split(',')[0] if ',' in location else location,
                        'state': 'Maharashtra',
                        'fees': college['metadata'].get('fees', 'Not Available'),
                        'placement_rating': college['metadata'].get('placement_rating', 'Not Available'),
                        'naac_rating': college.get('naac_rating', 'Not Available'),
                        'years_available': college['metadata'].get('years_available', []),
                        'trend_analysis': branch.get('trend_analysis', {}),
                        'source_url': f"DTE Maharashtra - {college['metadata'].get('last_updated', '')}"
                    }
                    records.append(record)
        
        return records


async def main():
    """Main execution function"""
    scraper = MHTCETScraper(years_back=5)
    
    try:
        # Run the full pipeline
        output_path = await scraper.run_full_pipeline()
        
        # Also create vector store compatible format
        with open(output_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        vector_records = scraper.create_vector_store_compatible_format(structured_data)
        
        # Save vector store compatible format
        vector_path = scraper.data_dir / "mht_cet_data.json"
        with open(vector_path, 'w', encoding='utf-8') as f:
            json.dump(vector_records, f, indent=2, ensure_ascii=False)
        
        scraper.logger.info(f"Vector store compatible data saved to: {vector_path}")
        scraper.logger.info(f"Total records for vector store: {len(vector_records)}")
        
        print(f"\n✅ Scraping completed successfully!")
        print(f"📁 Structured data: {output_path}")
        print(f"📁 Vector store data: {vector_path}")
        print(f"📊 Total colleges: {structured_data['metadata']['total_colleges']}")
        print(f"📊 Total records: {len(vector_records)}")
        print(f"📅 Years processed: {structured_data['metadata']['years_processed']}")
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        raise


if __name__ == "__main__":
    # Run the scraper
    asyncio.run(main())
