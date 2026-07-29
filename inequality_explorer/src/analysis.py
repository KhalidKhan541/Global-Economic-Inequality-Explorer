import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging


class InequalityAnalyzer:
    """Statistical analysis of economic inequality data."""

    def __init__(self, gini_df: pd.DataFrame, gdp_df: pd.DataFrame, hdi_df: pd.DataFrame,
                 quintile_df: pd.DataFrame, pop_df: pd.DataFrame):
        self.gini = gini_df
        self.gdp = gdp_df
        self.hdi = hdi_df
        self.quintiles = quintile_df
        self.population = pop_df
        self.logger = logging.getLogger(__name__)

    def gini_summary_stats(self) -> pd.DataFrame:
        """Summary statistics for Gini coefficients by region and year."""
        return self.gini.groupby(['region', 'year'])['gini_coefficient'].agg(
            ['mean', 'median', 'std', 'min', 'max', 'count']
        ).reset_index()

    def global_gini_trend(self) -> pd.DataFrame:
        """Population-weighted global Gini trend over time."""
        merged = self.gini.merge(self.population[['country_code', 'year', 'population']],
                                  on=['country_code', 'year'], how='left')
        merged['weighted_gini'] = merged['gini_coefficient'] * merged['population']

        trend = merged.groupby('year').apply(
            lambda x: pd.Series({
                'weighted_avg_gini': x['weighted_gini'].sum() / x['population'].sum(),
                'unweighted_avg_gini': x['gini_coefficient'].mean(),
                'std_gini': x['gini_coefficient'].std(),
                'country_count': x['country_code'].nunique()
            })
        ).reset_index()

        return trend

    def correlation_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Correlation matrix between key indicators."""
        gini_pivot = self.gini.pivot_table(index='country_code', columns='year',
                                            values='gini_coefficient', aggfunc='first')
        gdp_pivot = self.gdp.pivot_table(index='country_code', columns='year',
                                          values='gdp_per_capita', aggfunc='first')
        hdi_pivot = self.hdi.pivot_table(index='country_code', columns='year',
                                          values='hdi', aggfunc='first')

        latest_year = max(self.gini['year'].dropna())

        indicators = pd.DataFrame({
            'gini': gini_pivot[latest_year] if latest_year in gini_pivot.columns else np.nan,
            'gdp_per_capita': gdp_pivot[latest_year] if latest_year in gdp_pivot.columns else np.nan,
            'hdi': hdi_pivot[latest_year] if latest_year in hdi_pivot.columns else np.nan,
        }).dropna()

        corr_matrix = indicators.corr()

        pval_matrix = pd.DataFrame(np.zeros_like(corr_matrix),
                                    index=corr_matrix.index, columns=corr_matrix.columns)
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if i != j:
                    _, pval = stats.pearsonr(indicators[i].dropna(), indicators[j].dropna())
                    pval_matrix.loc[i, j] = pval

        return corr_matrix, pval_matrix

    def regional_inequality_trends(self) -> pd.DataFrame:
        """Inequality trends by region."""
        return self.gini.groupby(['region', 'year'])['gini_coefficient'].mean().reset_index()

    def top_bottom_countries(self, year: int = None, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Top and bottom countries by Gini coefficient."""
        if year is None:
            year = self.gini['year'].max()

        year_data = self.gini[self.gini['year'] == year].dropna(subset=['gini_coefficient'])

        top = year_data.nlargest(top_n, 'gini_coefficient')[['country_code', 'country_name', 'region', 'gini_coefficient']]
        bottom = year_data.nsmallest(top_n, 'gini_coefficient')[['country_code', 'country_name', 'region', 'gini_coefficient']]

        return top, bottom

    def gdp_gini_regression(self) -> Dict:
        """Regression analysis: Gini vs GDP per capita."""
        merged = self.gini.merge(self.gdp[['country_code', 'year', 'gdp_per_capita']],
                                  on=['country_code', 'year'], how='inner').dropna()

        if len(merged) < 10:
            return {'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan}

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(merged['gdp_per_capita']), merged['gini_coefficient']
        )

        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'n_observations': len(merged)
        }

    def hdi_gini_scatter_data(self, year: int = None) -> pd.DataFrame:
        """Data for HDI vs Gini scatter plot."""
        if year is None:
            year = min(self.gini['year'].max(), self.hdi['year'].max())

        gini_year = self.gini[self.gini['year'] == year][['country_code', 'country_name', 'region', 'gini_coefficient']]
        hdi_year = self.hdi[self.hdi['year'] == year][['country_code', 'hdi']]
        gdp_year = self.gdp[self.gdp['year'] == year][['country_code', 'gdp_per_capita']]
        pop_year = self.population[self.population['year'] == year][['country_code', 'population']]

        scatter = gini_year.merge(hdi_year, on='country_code', how='inner') \
                          .merge(gdp_year, on='country_code', how='inner') \
                          .merge(pop_year, on='country_code', how='left')

        return scatter.dropna()

    def quintile_distribution_summary(self) -> pd.DataFrame:
        """Summary of income quintile distributions."""
        return self.quintiles.groupby(['region', 'year']).agg({
            'q1_lowest': 'mean',
            'q2': 'mean',
            'q3_middle': 'mean',
            'q4': 'mean',
            'q5_highest': 'mean'
        }).reset_index()

    def convergence_analysis(self) -> Dict:
        """Beta convergence: do poorer countries grow faster?"""
        gdp_sorted = self.gdp.sort_values(['country_code', 'year'])
        gdp_sorted['gdp_growth'] = gdp_sorted.groupby('country_code')['gdp_per_capita'].pct_change()

        latest = gdp_sorted['year'].max()
        initial = gdp_sorted[gdp_sorted['year'] == latest - 10][['country_code', 'gdp_per_capita']].rename(
            columns={'gdp_per_capita': 'initial_gdp'}
        )
        growth = gdp_sorted[gdp_sorted['year'] == latest][['country_code', 'gdp_growth']]

        merged = initial.merge(growth, on='country_code', how='inner').dropna()

        if len(merged) < 10:
            return pd.DataFrame()

        slope, intercept, r_value, p_value, _ = stats.linregress(
            np.log(merged['initial_gdp']), merged['gdp_growth']
        )

        return {
            'convergence_coefficient': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'n_countries': len(merged)
        }

    def full_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run full statistical analysis."""
        self.logger.info("Running full inequality analysis...")
        results = {
            'summary_stats': self.gini_summary_stats(),
            'global_trend': self.global_gini_trend(),
            'regional_trends': self.regional_inequality_trends(),
            'quintile_summary': self.quintile_distribution_summary(),
            'hdi_gini_scatter': self.hdi_gini_scatter_data(),
        }

        corr, pvals = self.correlation_analysis()
        results['correlation'] = corr
        results['p_values'] = pvals
        results['regression'] = pd.DataFrame([self.gdp_gini_regression()])

        top, bottom = self.top_bottom_countries()
        results['top_inequality'] = top
        results['bottom_inequality'] = bottom

        self.logger.info("Analysis complete")
        return results
