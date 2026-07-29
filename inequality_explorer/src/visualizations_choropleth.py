import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
import logging

class ChoroplethVisualizations:
    """Create choropleth and map visualizations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def animated_gini_choropleth(self, df: pd.DataFrame, 
                                  color_scale: str = 'RdYlGn_r') -> go.Figure:
        """Animated choropleth map of Gini coefficients over time.
        Shows inequality changes 1990-2023 with play/pause button.
        """
        df_clean = df.dropna(subset=['gini_coefficient'])
        
        fig = px.choropleth(
            df_clean,
            locations='country_code',
            color='gini_coefficient',
            hover_name='country_name',
            hover_data={'gini_coefficient': ':.2f', 'region': True},
            animation_frame='year',
            color_continuous_scale=color_scale,
            range_color=[20, 70],
            title='Global Gini Coefficient (1990-2023)',
            labels={'gini_coefficient': 'Gini Index', 'country_code': 'Country'},
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
            ),
            coloraxis_colorbar=dict(
                title='Gini Index',
                tickvals=[25, 35, 45, 55, 65],
                ticktext=['Low (25)', 'Moderate (35)', 'High (45)', 'Very High (55)', 'Extreme (65)'],
            ),
            title_x=0.5,
            width=1200,
            height=700,
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {'prefix': 'Year: ', 'font': {'size': 16}},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.05,
            }],
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'y': 0,
                'x': 0.05,
                'xanchor': 'right',
                'yanchor': 'top',
                'pad': {'t': 60, 'r': 10},
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                    }]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                    }]},
                ],
            }],
        )
        
        return fig
    
    def gini_treemap(self, df: pd.DataFrame, year: int = None) -> go.Figure:
        """Treemap of Gini coefficients by region and country."""
        if year is None:
            year = df['year'].max()
        
        df_year = df[df['year'] == year].dropna(subset=['gini_coefficient'])
        
        fig = px.treemap(
            df_year,
            path=['region', 'country_name'],
            values='gini_coefficient',
            color='gini_coefficient',
            color_continuous_scale='RdYlGn_r',
            title=f'Income Inequality Treemap ({year})',
        )
        
        fig.update_layout(width=1000, height=700)
        return fig
    
    def gini_geographical_bubble(self, df: pd.DataFrame, gdp_df: pd.DataFrame,
                                  pop_df: pd.DataFrame, year: int = None) -> go.Figure:
        """Bubble map: bubble size = population, color = Gini, x = longitude proxy."""
        if year is None:
            year = df['year'].max()
        
        # Country centroids (approximate)
        CENTROIDS = {
            'USA': (39.8, -98.6), 'CHN': (35.9, 104.2), 'IND': (20.6, 78.9),
            'DEU': (51.2, 10.4), 'GBR': (55.4, -3.4), 'FRA': (46.2, 2.2),
            'JPN': (36.2, 138.3), 'BRA': (-14.2, -51.9), 'ZAF': (-30.6, 22.9),
            'NGA': (9.1, 8.7), 'RUS': (61.5, 105.3), 'MEX': (23.6, -102.6),
            'IDN': (-0.8, 113.9), 'TUR': (38.9, 35.2), 'ARG': (-38.4, -63.6),
            'AUS': (-25.3, 133.8), 'CAN': (56.1, -106.3), 'KOR': (35.9, 127.8),
            'ESP': (40.5, -3.7), 'ITA': (41.9, 12.6), 'POL': (51.9, 19.1),
            'THA': (15.9, 100.9), 'COL': (4.6, -74.3), 'KEN': (-0.02, 37.9),
            'EGY': (26.8, 30.8), 'SAU': (23.9, 45.1), 'CHL': (-35.7, -71.5),
            'PER': (-9.2, -75.0), 'MYS': (4.2, 101.9), 'PHL': (12.9, 121.8),
            'VNM': (14.1, 108.3), 'BGD': (23.7, 90.4), 'PAK': (30.4, 69.3),
            'ETH': (9.1, 40.5), 'GHA': (7.9, -1.0), 'TZA': (-6.4, 34.9),
            'MAR': (31.8, -7.1), 'IRQ': (33.2, 43.7), 'IRN': (32.4, 53.7),
        }
        
        df_year = df[df['year'] == year].dropna(subset=['gini_coefficient'])
        gdp_year = gdp_df[gdp_df['year'] == year][['country_code', 'gdp_per_capita']]
        pop_year = pop_df[pop_df['year'] == year][['country_code', 'population']]
        
        merged = df_year.merge(gdp_year, on='country_code', how='left') \
                       .merge(pop_year, on='country_code', how='left')
        
        # Add coordinates
        merged['lat'] = merged['country_code'].map(lambda x: CENTROIDS.get(x, (0, 0))[0])
        merged['lon'] = merged['country_code'].map(lambda x: CENTROIDS.get(x, (0, 0))[1])
        
        fig = px.scatter_geo(
            merged,
            lat='lat',
            lon='lon',
            size='population',
            color='gini_coefficient',
            hover_name='country_name',
            hover_data={'gini_coefficient': ':.2f', 'gdp_per_capita': ':,.0f', 'population': ':,.0f'},
            color_continuous_scale='RdYlGn_r',
            size_max=50,
            title=f'Global Inequality Map ({year})',
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
            ),
            width=1200,
            height=700,
        )
        
        return fig
    
    def regional_box_plot(self, df: pd.DataFrame) -> go.Figure:
        """Box plot of Gini distribution by region."""
        fig = px.box(
            df.dropna(subset=['gini_coefficient']),
            x='region',
            y='gini_coefficient',
            color='region',
            title='Gini Coefficient Distribution by Region',
            labels={'gini_coefficient': 'Gini Index', 'region': 'Region'},
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            width=1000,
            height=600,
        )
        
        return fig
    
    def inequality_gap_chart(self, df: pd.DataFrame) -> go.Figure:
        """Chart showing gap between most and least equal countries over time."""
        yearly_stats = df.groupby('year')['gini_coefficient'].agg([
            ('max_gini', 'max'),
            ('min_gini', 'min'),
            ('mean_gini', 'mean'),
        ]).reset_index()
        yearly_stats['gap'] = yearly_stats['max_gini'] - yearly_stats['min_gini']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Inequality Gap (Max - Min Gini)', 'Average Global Gini'),
            vertical_spacing=0.15,
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'], y=yearly_stats['gap'],
                      mode='lines+markers', name='Gap',
                      line=dict(color='crimson', width=2)),
            row=1, col=1,
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['year'], y=yearly_stats['mean_gini'],
                      mode='lines+markers', name='Mean Gini',
                      line=dict(color='steelblue', width=2),
                      fill='tozeroy'),
            row=2, col=1,
        )
        
        fig.update_layout(height=800, width=1000, title_text='Global Inequality Trends')
        return fig
