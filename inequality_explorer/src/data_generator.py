"""Synthetic World Bank-style economic data generator.

Generates realistic synthetic datasets covering income inequality,
GDP per capita, HDI, population, and life expectancy for 100+
countries across multiple regions, spanning 1990–2023.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class WorldBankDataGenerator:
    """Generate realistic synthetic World Bank economic data.

    Parameters
    ----------
    start_year : int
        First year of the generated time series (inclusive).
    end_year : int
        Last year of the generated time series (inclusive).
    seed : int
        Random seed for reproducibility.
    """

    COUNTRIES: Dict[str, Tuple[str, str]] = {
        'USA': ('United States', 'North America'), 'CHN': ('China', 'East Asia'),
        'IND': ('India', 'South Asia'), 'DEU': ('Germany', 'Europe'),
        'GBR': ('United Kingdom', 'Europe'), 'FRA': ('France', 'Europe'),
        'JPN': ('Japan', 'East Asia'), 'BRA': ('Brazil', 'Latin America'),
        'ZAF': ('South Africa', 'Sub-Saharan Africa'), 'NGA': ('Nigeria', 'Sub-Saharan Africa'),
        'RUS': ('Russia', 'Central Asia'), 'MEX': ('Mexico', 'Latin America'),
        'IDN': ('Indonesia', 'Southeast Asia'), 'TUR': ('Turkey', 'Middle East'),
        'SAU': ('Saudi Arabia', 'Middle East'), 'ARG': ('Argentina', 'Latin America'),
        'AUS': ('Australia', 'Oceania'), 'CAN': ('Canada', 'North America'),
        'KOR': ('South Korea', 'East Asia'), 'ESP': ('Spain', 'Europe'),
        'ITA': ('Italy', 'Europe'), 'POL': ('Poland', 'Europe'),
        'THA': ('Thailand', 'Southeast Asia'), 'COL': ('Colombia', 'Latin America'),
        'KEN': ('Kenya', 'Sub-Saharan Africa'), 'EGY': ('Egypt', 'Middle East'),
        'GHA': ('Ghana', 'Sub-Saharan Africa'), 'ETH': ('Ethiopia', 'Sub-Saharan Africa'),
        'BGD': ('Bangladesh', 'South Asia'), 'PAK': ('Pakistan', 'South Asia'),
        'VNM': ('Vietnam', 'Southeast Asia'), 'PHL': ('Philippines', 'Southeast Asia'),
        'MYS': ('Malaysia', 'Southeast Asia'), 'PER': ('Peru', 'Latin America'),
        'CHL': ('Chile', 'Latin America'), 'ECU': ('Ecuador', 'Latin America'),
        'MAR': ('Morocco', 'North Africa'), 'TUN': ('Tunisia', 'North Africa'),
        'DZA': ('Algeria', 'North Africa'), 'IRQ': ('Iraq', 'Middle East'),
        'IRN': ('Iran', 'Middle East'), 'ISR': ('Israel', 'Middle East'),
        'ARE': ('UAE', 'Middle East'), 'QAT': ('Qatar', 'Middle East'),
        'KWT': ('Kuwait', 'Middle East'), 'OMN': ('Oman', 'Middle East'),
        'JOR': ('Jordan', 'Middle East'), 'LBN': ('Lebanon', 'Middle East'),
        'SYR': ('Syria', 'Middle East'), 'YEM': ('Yemen', 'Middle East'),
        'AFG': ('Afghanistan', 'South Asia'), 'NPL': ('Nepal', 'South Asia'),
        'LKA': ('Sri Lanka', 'South Asia'), 'BTN': ('Bhutan', 'South Asia'),
        'MNG': ('Mongolia', 'East Asia'), 'MMR': ('Myanmar', 'Southeast Asia'),
        'KHM': ('Cambodia', 'Southeast Asia'), 'LAO': ('Laos', 'Southeast Asia'),
        'SGP': ('Singapore', 'Southeast Asia'), 'HKG': ('Hong Kong', 'East Asia'),
        'TWN': ('Taiwan', 'East Asia'), 'PRY': ('Paraguay', 'Latin America'),
        'BOL': ('Bolivia', 'Latin America'), 'URY': ('Uruguay', 'Latin America'),
        'GTM': ('Guatemala', 'Latin America'), 'HND': ('Honduras', 'Latin America'),
        'SLV': ('El Salvador', 'Latin America'), 'NIC': ('Nicaragua', 'Latin America'),
        'CRI': ('Costa Rica', 'Latin America'), 'PAN': ('Panama', 'Latin America'),
        'DOM': ('Dominican Republic', 'Latin America'), 'CUB': ('Cuba', 'Latin America'),
        'VEN': ('Venezuela', 'Latin America'), 'TTO': ('Trinidad and Tobago', 'Latin America'),
        'NAM': ('Namibia', 'Sub-Saharan Africa'), 'BWA': ('Botswana', 'Sub-Saharan Africa'),
        'LSO': ('Lesotho', 'Sub-Saharan Africa'), 'SWZ': ('Eswatini', 'Sub-Saharan Africa'),
        'AGO': ('Angola', 'Sub-Saharan Africa'), 'ZMB': ('Zambia', 'Sub-Saharan Africa'),
        'ZWE': ('Zimbabwe', 'Sub-Saharan Africa'), 'MWI': ('Malawi', 'Sub-Saharan Africa'),
        'MOZ': ('Mozambique', 'Sub-Saharan Africa'), 'MDG': ('Madagascar', 'Sub-Saharan Africa'),
        'RWA': ('Rwanda', 'Sub-Saharan Africa'), 'BDI': ('Burundi', 'Sub-Saharan Africa'),
        'COD': ('DR Congo', 'Sub-Saharan Africa'), 'TZA': ('Tanzania', 'Sub-Saharan Africa'),
        'UGA': ('Uganda', 'Sub-Saharan Africa'), 'SEN': ('Senegal', 'Sub-Saharan Africa'),
        'MLI': ('Mali', 'Sub-Saharan Africa'), 'BFA': ('Burkina Faso', 'Sub-Saharan Africa'),
        'NER': ('Niger', 'Sub-Saharan Africa'), 'GIN': ('Guinea', 'Sub-Saharan Africa'),
        'SLE': ('Sierra Leone', 'Sub-Saharan Africa'), 'LBR': ('Liberia', 'Sub-Saharan Africa'),
        'TGO': ('Togo', 'Sub-Saharan Africa'), 'BEN': ('Benin', 'Sub-Saharan Africa'),
        'CMR': ('Cameroon', 'Sub-Saharan Africa'), 'CIV': ('Ivory Coast', 'Sub-Saharan Africa'),
        'GAB': ('Gabon', 'Sub-Saharan Africa'), 'COG': ('Congo', 'Sub-Saharan Africa'),
        'GNQ': ('Equatorial Guinea', 'Sub-Saharan Africa'),
        'MUS': ('Mauritius', 'Sub-Saharan Africa'),
    }

    REGION_PARAMS: Dict[str, Dict[str, float]] = {
        'North America':        {'gini_base': 35, 'gdp_base': 55000, 'hdi_base': 0.92, 'gini_trend': -0.05},
        'Europe':               {'gini_base': 31, 'gdp_base': 40000, 'hdi_base': 0.88, 'gini_trend': -0.03},
        'East Asia':            {'gini_base': 38, 'gdp_base': 25000, 'hdi_base': 0.78, 'gini_trend': 0.02},
        'South Asia':           {'gini_base': 35, 'gdp_base': 3000,  'hdi_base': 0.60, 'gini_trend': 0.01},
        'Southeast Asia':       {'gini_base': 38, 'gdp_base': 8000,  'hdi_base': 0.70, 'gini_trend': 0.0},
        'Latin America':        {'gini_base': 48, 'gdp_base': 12000, 'hdi_base': 0.72, 'gini_trend': -0.08},
        'Sub-Saharan Africa':   {'gini_base': 44, 'gdp_base': 2500,  'hdi_base': 0.52, 'gini_trend': 0.03},
        'Middle East':          {'gini_base': 37, 'gdp_base': 18000, 'hdi_base': 0.72, 'gini_trend': 0.01},
        'North Africa':         {'gini_base': 36, 'gdp_base': 6000,  'hdi_base': 0.68, 'gini_trend': 0.0},
        'Central Asia':         {'gini_base': 35, 'gdp_base': 10000, 'hdi_base': 0.74, 'gini_trend': -0.02},
        'Oceania':              {'gini_base': 34, 'gdp_base': 50000, 'hdi_base': 0.93, 'gini_trend': 0.01},
    }

    def __init__(self, start_year: int = 1990, end_year: int = 2023, seed: int = 42) -> None:
        self.start_year = start_year
        self.end_year = end_year
        self.years: List[int] = list(range(start_year, end_year + 1))
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)

    def _get_region(self, iso3: str) -> str:
        """Return the region associated with *iso3*."""
        return self.COUNTRIES.get(iso3, ('Unknown', 'Unknown'))[1]

    # ------------------------------------------------------------------
    # Individual generators
    # ------------------------------------------------------------------

    def generate_gini_coefficients(self) -> pd.DataFrame:
        """Generate Gini coefficients for every country across all years.

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            gini_coefficient (float | None).
        """
        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
            base_gini = params['gini_base']
            trend = params['gini_trend']

            country_offset = np.random.uniform(-8, 8)
            gini = base_gini + country_offset

            for i, year in enumerate(self.years):
                gini_val = gini + (trend * i) + np.random.normal(0, 1.2)
                gini_val = float(np.clip(gini_val, 20, 70))

                if np.random.random() < 0.05:
                    gini_val = None
                else:
                    gini_val = round(gini_val, 2)

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'gini_coefficient': gini_val,
                })

        return pd.DataFrame(records)

    def generate_income_quintiles(self) -> pd.DataFrame:
        """Generate income-share-by-quintile data.

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            q1_lowest, q2, q3_middle, q4, q5_highest (all float).
        """
        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
            base_gini = params['gini_base']

            inequality_factor = (base_gini - 25) / 45  # 0 = equal, 1 = very unequal

            for year in self.years:
                q1 = 20 - (inequality_factor * 14) + np.random.normal(0, 0.5)
                q2 = 20 - (inequality_factor * 8) + np.random.normal(0, 0.5)
                q3 = 20 - (inequality_factor * 2) + np.random.normal(0, 0.5)
                q5 = 20 + (inequality_factor * 30) + np.random.normal(0, 1)
                q4 = 100 - q1 - q2 - q3 - q5

                shares = np.array([q1, q2, q3, q4, q5])
                shares = np.clip(shares, 1, 80)
                shares = shares / shares.sum() * 100

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'q1_lowest': round(float(shares[0]), 2),
                    'q2': round(float(shares[1]), 2),
                    'q3_middle': round(float(shares[2]), 2),
                    'q4': round(float(shares[3]), 2),
                    'q5_highest': round(float(shares[4]), 2),
                })

        return pd.DataFrame(records)

    def generate_gdp_per_capita(self) -> pd.DataFrame:
        """Generate GDP per capita (constant 2015 USD).

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            gdp_per_capita (float | None).
        """
        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
            base_gdp = params['gdp_base']
            country_offset = np.random.uniform(0.5, 2.0)
            gdp = base_gdp * country_offset

            for i, year in enumerate(self.years):
                growth_rate = (
                    0.02 if base_gdp > 30000 else 0.04 + np.random.normal(0, 0.01)
                )
                gdp_val = gdp * ((1 + growth_rate) ** i) + np.random.normal(0, gdp * 0.02)
                gdp_val = max(gdp_val, 200)

                if np.random.random() < 0.03:
                    gdp_val = None
                else:
                    gdp_val = round(float(gdp_val), 2)

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'gdp_per_capita': gdp_val,
                })

        return pd.DataFrame(records)

    def generate_hdi(self) -> pd.DataFrame:
        """Generate Human Development Index values.

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            hdi (float | None).
        """
        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
            base_hdi = params['hdi_base']
            country_offset = np.random.uniform(-0.08, 0.08)
            hdi = base_hdi + country_offset

            for i, year in enumerate(self.years):
                hdi_val = hdi + (i * 0.003) + np.random.normal(0, 0.005)
                hdi_val = float(np.clip(hdi_val, 0.25, 0.98))

                if np.random.random() < 0.04:
                    hdi_val = None
                else:
                    hdi_val = round(hdi_val, 4)

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'hdi': hdi_val,
                })

        return pd.DataFrame(records)

    def generate_population(self) -> pd.DataFrame:
        """Generate population estimates.

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            population (int).
        """
        pop_bases: Dict[str, int] = {
            'CHN': 1_150_000_000, 'IND': 900_000_000, 'USA': 250_000_000,
            'IDN': 180_000_000, 'BRA': 150_000_000, 'PAK': 120_000_000,
            'BGD': 110_000_000, 'NGA': 100_000_000, 'RUS': 148_000_000,
            'JPN': 124_000_000, 'MEX': 90_000_000, 'PHL': 65_000_000,
            'VNM': 70_000_000, 'ETH': 55_000_000, 'EGY': 60_000_000,
            'DEU': 82_000_000, 'GBR': 57_000_000, 'FRA': 58_000_000,
            'ITA': 57_000_000, 'TUR': 60_000_000, 'ZAF': 45_000_000,
            'TZA': 30_000_000, 'KEN': 30_000_000, 'COL': 38_000_000,
            'POL': 38_000_000, 'CAN': 30_000_000, 'MAR': 28_000_000,
            'SAU': 22_000_000, 'PER': 25_000_000, 'MYS': 22_000_000,
            'ARG': 37_000_000, 'UKR': 52_000_000,
        }

        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            base_pop = pop_bases.get(iso3, int(np.random.randint(1_000_000, 50_000_000)))
            pop = base_pop

            for i, year in enumerate(self.years):
                params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
                growth = (
                    0.01
                    if params['gdp_base'] > 30_000
                    else 0.025 + np.random.normal(0, 0.005)
                )
                pop_val = int(pop * ((1 + growth) ** i))

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'population': pop_val,
                })

        return pd.DataFrame(records)

    def generate_life_expectancy(self) -> pd.DataFrame:
        """Generate life expectancy at birth.

        Returns
        -------
        pd.DataFrame
            Columns: country_code, country_name, region, year,
            life_expectancy (float | None).
        """
        records: List[dict] = []
        for iso3, (name, region) in self.COUNTRIES.items():
            params = self.REGION_PARAMS.get(region, self.REGION_PARAMS['Europe'])
            base_le = 55 + (params['hdi_base'] * 25)
            country_offset = np.random.uniform(-5, 5)
            le = base_le + country_offset

            for i, year in enumerate(self.years):
                le_val = le + (i * 0.15) + np.random.normal(0, 0.5)
                le_val = float(np.clip(le_val, 40, 88))

                if np.random.random() < 0.03:
                    le_val = None
                else:
                    le_val = round(le_val, 1)

                records.append({
                    'country_code': iso3,
                    'country_name': name,
                    'region': region,
                    'year': year,
                    'life_expectancy': le_val,
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate every dataset and return them keyed by name.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``'gini'``, ``'income_quintiles'``, ``'gdp'``,
            ``'hdi'``, ``'population'``, ``'life_expectancy'``.
        """
        self.logger.info("Generating synthetic World Bank data...")
        datasets: Dict[str, pd.DataFrame] = {
            'gini': self.generate_gini_coefficients(),
            'income_quintiles': self.generate_income_quintiles(),
            'gdp': self.generate_gdp_per_capita(),
            'hdi': self.generate_hdi(),
            'population': self.generate_population(),
            'life_expectancy': self.generate_life_expectancy(),
        }
        self.logger.info("Generated %d datasets", len(datasets))
        return datasets

    def save_to_csv(self, output_dir: str = 'data') -> None:
        """Generate all datasets and write each to a CSV file.

        Parameters
        ----------
        output_dir : str
            Directory in which to create ``<name>.csv`` for every
            generated dataset.  Created if it does not exist.
        """
        os.makedirs(output_dir, exist_ok=True)
        datasets = self.generate_all()
        for name, df in datasets.items():
            path = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(path, index=False)
            self.logger.info("Saved %s", path)
