import requests
import pandas as pd
import time
import logging
import os
from typing import List, Optional, Dict
from datetime import datetime


# Placeholder country dictionary — replace with actual World Bank country codes
WORLD_BANK_COUNTRIES = {
    'USA': 'United States',
    'CHN': 'China',
    'IND': 'India',
    'BRA': 'Brazil',
    'GBR': 'United Kingdom',
    'FRA': 'France',
    # ... add more as needed
}


class WorldBankAPIIngestion:
    """Ingest data from World Bank API."""
    
    BASE_URL = "https://api.worldbank.org/v2"
    
    # World Bank indicator codes
    INDICATORS = {
        'gini': 'SI.POV.GINI',
        'gdp_per_capita': 'NY.GDP.PCAP.KD',
        'hdi': 'HD.HDI.OVRL',  # Note: HDI not always in WB API, may need UNDP
        'population': 'SP.POP.TOTL',
        'life_expectancy': 'SP.DYN.LE00.IN',
        'poverty_rate': 'SI.POV.DDAY',
        'education_expenditure': 'SE.XPD.TOTL.GD.ZS',
        'health_expenditure': 'SH.XPD.CHEX.GD.ZS',
        'unemployment': 'SL.UEM.TOTL.ZS',
        'inflation': 'FP.CPI.TOTL.ZG',
        'trade_openness': 'NE.TRD.GNFS.ZS',
        'urban_population': 'SP.URB.TOTL.IN.ZS',
        'co2_emissions': 'EN.ATM.CO2E.PC',
        'internet_usage': 'IT.NET.USER.ZS',
        'mobile_subscriptions': 'IT.CEL.SETS.P2',
    }
    
    def __init__(self, cache_dir: str = 'data/api_cache'):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
    
    def _fetch_indicator(self, indicator_code: str, countries: List[str] = None,
                        start_year: int = 1990, end_year: int = 2023,
                        per_page: int = 1000, max_retries: int = 3) -> pd.DataFrame:
        """Fetch a single indicator from World Bank API.
        
        Handles pagination, rate limiting, and retries for failed requests.
        
        Args:
            indicator_code: World Bank indicator code (e.g., 'SI.POV.GINI')
            countries: List of country codes to fetch. Defaults to all known countries.
            start_year: Start year for data range
            end_year: End year for data range
            per_page: Number of records per API page
            max_retries: Maximum number of retry attempts for failed requests
            
        Returns:
            DataFrame with columns: country_code, country_name, year, value, indicator
        """
        if countries is None:
            countries = list(WORLD_BANK_COUNTRIES.keys())
        
        all_records = []
        page = 1
        total_pages = 1
        
        while page <= total_pages:
            params = {
                'format': 'json',
                'date': f'{start_year}:{end_year}',
                'per_page': per_page,
                'page': page,
            }
            
            country_str = ';'.join(countries)
            url = f"{self.BASE_URL}/country/{country_str}/indicator/{indicator_code}"
            
            retries = 0
            while retries < max_retries:
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if len(data) < 2:
                        break
                    
                    metadata = data[0]
                    total_pages = metadata.get('pages', 1)
                    
                    records = data[1]
                    if records is None:
                        break
                    
                    for record in records:
                        all_records.append({
                            'country_code': record.get('country', {}).get('id', ''),
                            'country_name': record.get('country', {}).get('value', ''),
                            'year': int(record.get('date', 0)),
                            'value': record.get('value'),
                            'indicator': indicator_code,
                        })
                    
                    break  # Success — exit retry loop
                    
                except requests.RequestException as e:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    self.logger.warning(
                        f"API request failed (attempt {retries}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            else:
                self.logger.error(
                    f"Failed to fetch {indicator_code} page {page} after {max_retries} retries"
                )
                break
            
            page += 1
            time.sleep(0.3)  # Rate limiting
        
        return pd.DataFrame(all_records) if all_records else pd.DataFrame()
    
    def fetch_all_indicators(self, countries: List[str] = None,
                            start_year: int = 1990, end_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """Fetch all configured indicators from the World Bank API.
        
        Fetches each indicator, caches results locally, and returns a dictionary
        mapping indicator names to DataFrames.
        
        Args:
            countries: List of country codes. Defaults to all known countries.
            start_year: Start year for data range
            end_year: End year for data range
            
        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        results = {}
        for name, code in self.INDICATORS.items():
            self.logger.info(f"Fetching {name} ({code})...")
            df = self._fetch_indicator(code, countries, start_year, end_year)
            if not df.empty:
                results[name] = df
                # Cache locally
                cache_path = os.path.join(self.cache_dir, f'{name}.csv')
                df.to_csv(cache_path, index=False)
        return results
    
    def load_from_cache(self) -> Dict[str, pd.DataFrame]:
        """Load cached data from local CSV files.
        
        Checks for previously cached indicator files and loads them into
        DataFrames.
        
        Returns:
            Dictionary mapping indicator names to cached DataFrames
        """
        results = {}
        for name in self.INDICATORS:
            cache_path = os.path.join(self.cache_dir, f'{name}.csv')
            if os.path.exists(cache_path):
                results[name] = pd.read_csv(cache_path)
        return results
    
    def pivot_to_wide(self, df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
        """Convert long format to wide (country x year) matrix.
        
        Reshapes a DataFrame from long format (one row per country-year-indicator)
        to wide format (one row per country, one column per year).
        
        Args:
            df: Long-format DataFrame with country_code, country_name, year, and value columns
            value_col: Name of the column containing values to pivot
            
        Returns:
            Wide-format DataFrame with country info as index and years as columns
        """
        if df.empty:
            return df
        return df.pivot_table(
            index=['country_code', 'country_name'],
            columns='year',
            values=value_col,
            aggfunc='first'
        ).reset_index()
