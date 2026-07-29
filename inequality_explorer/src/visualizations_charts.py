import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

class ChartVisualizations:
    """Create interactive chart visualizations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def income_quintile_stacked_bar(self, df: pd.DataFrame, countries: List[str] = None,
                                     year: int = None) -> go.Figure:
        """Stacked bar chart of income quintile distribution."""
        if year is None:
            year = df['year'].max()
        
        df_year = df[df['year'] == year]
        if countries:
            df_year = df_year[df_year['country_code'].isin(countries)]
        
        fig = go.Figure()
        
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
        quintiles = ['q1_lowest', 'q2', 'q3_middle', 'q4', 'q5_highest']
        labels = ['Q1 (Lowest 20%)', 'Q2', 'Q3 (Middle)', 'Q4', 'Q5 (Highest 20%)']
        
        for i, (q, label) in enumerate(zip(quintiles, labels)):
            fig.add_trace(go.Bar(
                name=label,
                x=df_year['country_name'],
                y=df_year[q],
                marker_color=colors[i],
                text=df_year[q].round(1),
                textposition='inside',
            ))
        
        fig.update_layout(
            barmode='stack',
            title=f'Income Distribution by Quintile ({year})',
            xaxis_title='Country',
            yaxis_title='Income Share (%)',
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            width=1200,
            height=600,
        )
        
        return fig
    
    def hdi_vs_gdp_bubble(self, df: pd.DataFrame, year: int = None) -> go.Figure:
        """Bubble chart: HDI vs GDP per capita, bubble size = population."""
        if year is None:
            year = df['year'].max()
        
        fig = px.scatter(
            df,
            x='gdp_per_capita',
            y='hdi',
            size='population',
            color='region',
            hover_name='country_name',
            hover_data={'gdp_per_capita': ':,.0f', 'hdi': ':.3f', 'population': ':,.0f'},
            title=f'HDI vs GDP per Capita ({year})',
            labels={
                'gdp_per_capita': 'GDP per Capita (USD)',
                'hdi': 'Human Development Index',
                'population': 'Population',
            },
            size_max=60,
            log_x=True,
        )
        
        # Add quadrant lines
        fig.add_hline(y=0.7, line_dash='dash', line_color='gray', opacity=0.5,
                      annotation_text='High HDI Threshold')
        fig.add_vline(x=12000, line_dash='dash', line_color='gray', opacity=0.5,
                      annotation_text='Upper-Middle Income')
        
        fig.update_layout(
            width=1100,
            height=700,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        )
        
        return fig
    
    def cross_country_scatter(self, x_data: pd.DataFrame, y_data: pd.DataFrame,
                               x_col: str, y_col: str, x_label: str, y_label: str,
                               year: int = None) -> go.Figure:
        """Cross-country scatter with play button animation."""
        if year is None:
            year = min(x_data['year'].max(), y_data['year'].max())
        
        merged = x_data.merge(y_data, on=['country_code', 'year', 'country_name'], how='inner')
        
        fig = px.scatter(
            merged,
            x=x_col,
            y=y_col,
            color='region',
            hover_name='country_name',
            animation_frame='year',
            animation_group='country_code',
            title=f'{x_label} vs {y_label} (1990-2023)',
            labels={x_col: x_label, y_col: y_label},
            range_x=[merged[x_col].min() * 0.9, merged[x_col].max() * 1.1],
            range_y=[merged[y_col].min() * 0.9, merged[y_col].max() * 1.1],
        )
        
        fig.update_layout(
            width=1100,
            height=700,
            updatemenus=[{
                'type': 'buttons',
                'showactive': True,
                'y': 0,
                'x': 0.05,
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
    
    def correlation_heatmap(self, corr_matrix: pd.DataFrame, pval_matrix: pd.DataFrame = None) -> go.Figure:
        """Correlation heatmap with significance annotations."""
        text_matrix = []
        for i in corr_matrix.index:
            row_text = []
            for j in corr_matrix.columns:
                val = corr_matrix.loc[i, j]
                if pval_matrix is not None:
                    pval = pval_matrix.loc[i, j]
                    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    row_text.append(f'{val:.3f}{stars}')
                else:
                    row_text.append(f'{val:.3f}')
            text_matrix.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            text=text_matrix,
            texttemplate='%{text}',
            textfont={'size': 12},
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
        ))
        
        fig.update_layout(
            title='Indicator Correlation Matrix (* p<0.05, ** p<0.01, *** p<0.001)',
            width=800,
            height=700,
            xaxis_tickangle=-45,
        )
        
        return fig
    
    def gini_distribution_histogram(self, df: pd.DataFrame, year: int = None) -> go.Figure:
        """Histogram of Gini coefficient distribution."""
        if year is None:
            year = df['year'].max()
        
        df_year = df[df['year'] == year].dropna(subset=['gini_coefficient'])
        
        fig = px.histogram(
            df_year,
            x='gini_coefficient',
            nbins=25,
            color='region',
            title=f'Distribution of Gini Coefficients ({year})',
            labels={'gini_coefficient': 'Gini Index', 'count': 'Number of Countries'},
            opacity=0.7,
        )
        
        fig.add_vline(x=df_year['gini_coefficient'].mean(), line_dash='dash', 
                      line_color='red', annotation_text='Mean')
        fig.add_vline(x=df_year['gini_coefficient'].median(), line_dash='dash',
                      line_color='blue', annotation_text='Median')
        
        fig.update_layout(width=1000, height=500)
        return fig
    
    def top_bottom_chart(self, top_df: pd.DataFrame, bottom_df: pd.DataFrame, 
                          year: int = None) -> go.Figure:
        """Horizontal bar chart of top and bottom countries by Gini."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Most Unequal Countries', 'Most Equal Countries'),
            horizontal_spacing=0.15,
        )
        
        # Most unequal (high Gini = red)
        fig.add_trace(go.Bar(
            y=top_df['country_name'],
            x=top_df['gini_coefficient'],
            orientation='h',
            marker_color='crimson',
            name='Most Unequal',
            text=top_df['gini_coefficient'].round(1),
            textposition='inside',
        ), row=1, col=1)
        
        # Most equal (low Gini = green)
        fig.add_trace(go.Bar(
            y=bottom_df['country_name'],
            x=bottom_df['gini_coefficient'],
            orientation='h',
            marker_color='forestgreen',
            name='Most Equal',
            text=bottom_df['gini_coefficient'].round(1),
            textposition='inside',
        ), row=1, col=2)
        
        fig.update_layout(
            title=f'Top & Bottom Countries by Inequality ({year})',
            showlegend=False,
            width=1200,
            height=500,
        )
        
        return fig
    
    def gini_trend_lines(self, df: pd.DataFrame, countries: List[str] = None) -> go.Figure:
        """Multi-line chart of Gini trends for selected countries."""
        if countries:
            df_filtered = df[df['country_code'].isin(countries)]
        else:
            # Default to some interesting countries
            default_countries = ['USA', 'CHN', 'BRA', 'ZAF', 'IND', 'DEU', 'NGA', 'JPN']
            df_filtered = df[df['country_code'].isin(default_countries)]
        
        fig = px.line(
            df_filtered,
            x='year',
            y='gini_coefficient',
            color='country_name',
            title='Gini Coefficient Trends by Country',
            labels={'gini_coefficient': 'Gini Index', 'year': 'Year'},
        )
        
        fig.update_layout(
            width=1100,
            height=600,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        )
        
        return fig
    
    def regional_radar_chart(self, df: pd.DataFrame, year: int = None) -> go.Figure:
        """Radar chart comparing regions on multiple dimensions."""
        if year is None:
            year = df['year'].max()
        
        # This would need normalized metrics for each region
        # Placeholder implementation
        regions = df['region'].unique()
        
        fig = go.Figure()
        
        fig.update_layout(
            title=f'Regional Comparison ({year})',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            width=800,
            height=600,
        )
        
        return fig
