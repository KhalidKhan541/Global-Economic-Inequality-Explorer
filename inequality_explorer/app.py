import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import os
import logging

# Import visualization modules
from src.visualizations_choropleth import ChoroplethVisualizations
from src.visualizations_charts import ChartVisualizations
from src.analysis import InequalityAnalyzer
from src.data_generator import WorldBankDataGenerator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load or generate data
DATA_DIR = 'data'
if os.path.exists(os.path.join(DATA_DIR, 'gini.csv')):
    logger.info("Loading cached data...")
    gini_df = pd.read_csv(os.path.join(DATA_DIR, 'gini.csv'))
    quintiles_df = pd.read_csv(os.path.join(DATA_DIR, 'income_quintiles.csv'))
    gdp_df = pd.read_csv(os.path.join(DATA_DIR, 'gdp.csv'))
    hdi_df = pd.read_csv(os.path.join(DATA_DIR, 'hdi.csv'))
    pop_df = pd.read_csv(os.path.join(DATA_DIR, 'population.csv'))
    le_df = pd.read_csv(os.path.join(DATA_DIR, 'life_expectancy.csv'))
else:
    logger.info("Generating synthetic data...")
    gen = WorldBankDataGenerator()
    data = gen.generate_all()
    gini_df = data['gini']
    quintiles_df = data['income_quintiles']
    gdp_df = data['gdp']
    hdi_df = data['hdi']
    pop_df = data['population']
    le_df = data['life_expectancy']
    gen.save_to_csv(DATA_DIR)

# Initialize visualizations
choropleth_viz = ChoroplethVisualizations()
chart_viz = ChartVisualizations()

# Merge data for analysis
scatter_data = hdi_df.merge(gini_df[['country_code', 'year', 'gini_coefficient']], 
                            on=['country_code', 'year'], how='inner')
scatter_data = scatter_data.merge(gdp_df[['country_code', 'year', 'gdp_per_capita']],
                                   on=['country_code', 'year'], how='inner')
scatter_data = scatter_data.merge(pop_df[['country_code', 'year', 'population']],
                                   on=['country_code', 'year'], how='left')

# Initialize Dash app
app = dash.Dash(__name__, title='Global Economic Inequality Explorer')
server = app.server

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Global Economic Inequality Explorer', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P('Interactive dashboard analyzing global income inequality patterns (1990-2023)',
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'}),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Tabs
    dcc.Tabs([
        # Tab 1: Global Overview
        dcc.Tab(label='Global Overview', children=[
            html.Div([
                html.Div([
                    html.H3('Animated Gini Coefficient Map'),
                    html.P('Watch how global inequality has evolved over time'),
                    dcc.Graph(id='gini-choropleth'),
                ], style={'width': '100%', 'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        html.H4('Gini Distribution by Region'),
                        dcc.Graph(id='regional-box'),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H4('Inequality Gap Over Time'),
                        dcc.Graph(id='gap-chart'),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                ]),
            ], style={'padding': '20px'}),
        ]),
        
        # Tab 2: Country Comparison
        dcc.Tab(label='Country Comparison', children=[
            html.Div([
                html.Div([
                    html.H3('Select Countries to Compare'),
                    dcc.Dropdown(
                        id='country-selector',
                        options=[{'label': row['country_name'], 'value': row['country_code']}
                                 for _, row in gini_df.drop_duplicates('country_code').iterrows()],
                        value=['USA', 'CHN', 'BRA', 'IND', 'DEU', 'ZAF'],
                        multi=True,
                        placeholder='Select countries...',
                    ),
                ], style={'width': '100%', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H4('Gini Coefficient Trends'),
                        dcc.Graph(id='gini-trends'),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H4('Income Quintile Distribution'),
                        dcc.Graph(id='quintile-chart'),
                        html.Div([
                            html.Label('Select Year:'),
                            dcc.Slider(
                                id='year-slider-quintile',
                                min=gini_df['year'].min(),
                                max=gini_df['year'].max(),
                                value=gini_df['year'].max(),
                                marks={str(y): str(y) for y in range(gini_df['year'].min(), gini_df['year'].max()+1, 5)},
                                step=None,
                            ),
                        ]),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                ]),
            ], style={'padding': '20px'}),
        ]),
        
        # Tab 3: Development Analysis
        dcc.Tab(label='Development Analysis', children=[
            html.Div([
                html.Div([
                    html.H3('HDI vs GDP per Capita'),
                    html.P('Bubble size represents population'),
                    dcc.Graph(id='hdi-gdp-bubble'),
                ], style={'width': '100%', 'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        html.H4('Correlation Heatmap'),
                        dcc.Graph(id='correlation-heatmap'),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H4('Top & Bottom Countries'),
                        dcc.Graph(id='top-bottom-chart'),
                        html.Div([
                            html.Label('Select Year:'),
                            dcc.Slider(
                                id='year-slider-rank',
                                min=gini_df['year'].min(),
                                max=gini_df['year'].max(),
                                value=gini_df['year'].max(),
                                marks={str(y): str(y) for y in range(gini_df['year'].min(), gini_df['year'].max()+1, 5)},
                                step=None,
                            ),
                        ]),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                ]),
            ], style={'padding': '20px'}),
        ]),
        
        # Tab 4: Cross-Country Scatter
        dcc.Tab(label='Cross-Country Analysis', children=[
            html.Div([
                html.Div([
                    html.H3('Cross-Country Animated Scatter'),
                    html.Div([
                        html.Label('X-Axis:'),
                        dcc.Dropdown(
                            id='x-axis-selector',
                            options=[
                                {'label': 'GDP per Capita', 'value': 'gdp_per_capita'},
                                {'label': 'HDI', 'value': 'hdi'},
                                {'label': 'Life Expectancy', 'value': 'life_expectancy'},
                            ],
                            value='gdp_per_capita',
                        ),
                        html.Label('Y-Axis:'),
                        dcc.Dropdown(
                            id='y-axis-selector',
                            options=[
                                {'label': 'Gini Coefficient', 'value': 'gini_coefficient'},
                                {'label': 'HDI', 'value': 'hdi'},
                                {'label': 'Life Expectancy', 'value': 'life_expectancy'},
                            ],
                            value='gini_coefficient',
                        ),
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='cross-country-scatter'),
                ], style={'width': '100%'}),
            ], style={'padding': '20px'}),
        ]),
        
        # Tab 5: Statistics
        dcc.Tab(label='Statistics', children=[
            html.Div([
                html.H3('Statistical Summary'),
                html.Div(id='stats-summary', children=[
                    html.P('Select metrics to view statistical analysis.'),
                ]),
            ], style={'padding': '20px'}),
        ]),
    ]),
    
    # Footer
    html.Div([
        html.P('Data: Synthetic World Bank-style data | Built with Plotly Dash',
               style={'textAlign': 'center', 'color': '#95a5a6'}),
    ], style={'marginTop': '30px', 'padding': '10px', 'backgroundColor': '#ecf0f1'}),
])

# Callbacks
@callback(
    Output('gini-choropleth', 'figure'),
    Input('gini-choropleth', 'id')
)
def update_choropleth(_):
    return choropleth_viz.animated_gini_choropleth(gini_df)

@callback(
    Output('regional-box', 'figure'),
    Input('regional-box', 'id')
)
def update_regional_box(_):
    return choropleth_viz.regional_box_plot(gini_df)

@callback(
    Output('gap-chart', 'figure'),
    Input('gap-chart', 'id')
)
def update_gap_chart(_):
    return choropleth_viz.inequality_gap_chart(gini_df)

@callback(
    Output('gini-trends', 'figure'),
    Input('country-selector', 'value')
)
def update_gini_trends(selected_countries):
    return chart_viz.gini_trend_lines(gini_df, selected_countries)

@callback(
    Output('quintile-chart', 'figure'),
    Input('country-selector', 'value'),
    Input('year-slider-quintile', 'value')
)
def update_quintile_chart(selected_countries, year):
    return chart_viz.income_quintile_stacked_bar(quintiles_df, selected_countries, year)

@callback(
    Output('hdi-gdp-bubble', 'figure'),
    Input('hdi-gdp-bubble', 'id')
)
def update_hdi_gdp(_):
    return chart_viz.hdi_vs_gdp_bubble(scatter_data)

@callback(
    Output('correlation-heatmap', 'figure'),
    Input('correlation-heatmap', 'id')
)
def update_correlation(_):
    analyzer = InequalityAnalyzer(gini_df, gdp_df, hdi_df, quintiles_df, pop_df)
    corr, pvals = analyzer.correlation_analysis()
    return chart_viz.correlation_heatmap(corr, pvals)

@callback(
    Output('top-bottom-chart', 'figure'),
    Input('year-slider-rank', 'value')
)
def update_top_bottom(year):
    analyzer = InequalityAnalyzer(gini_df, gdp_df, hdi_df, quintiles_df, pop_df)
    top, bottom = analyzer.top_bottom_countries(year)
    return chart_viz.top_bottom_chart(top, bottom, year)

@callback(
    Output('cross-country-scatter', 'figure'),
    Input('x-axis-selector', 'value'),
    Input('y-axis-selector', 'value')
)
def update_scatter(x_col, y_col):
    x_label = x_col.replace('_', ' ').title()
    y_label = y_col.replace('_', ' ').title()
    return chart_viz.cross_country_scatter(scatter_data, scatter_data, x_col, y_col, x_label, y_label)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
