# Global Economic Inequality Explorer

Interactive Plotly Dash dashboard for visualizing global economic inequality patterns, featuring animated choropleth maps, income distribution charts, and statistical analysis.

## Features

### Visualizations
- **Animated Choropleth Map**: Gini coefficient changes 1990-2023 with play/pause controls
- **Income Quintile Distribution**: Stacked bar charts showing income share by quintile
- **HDI vs GDP Bubble Chart**: Development indicators with population-weighted bubbles
- **Correlation Heatmap**: Statistical relationships between indicators with significance levels
- **Cross-Country Scatter**: Animated scatter plots with play button
- **Regional Box Plots**: Distribution comparison across regions
- **Treemap**: Hierarchical view of inequality by region and country

### Data Pipeline
- World Bank API ingestion with caching
- Synthetic data generator for offline use
- Statistical analysis layer with regression and correlation

### Dashboard Tabs
1. **Global Overview**: Animated map, regional distributions, inequality trends
2. **Country Comparison**: Multi-country trend analysis, quintile distributions
3. **Development Analysis**: HDI vs GDP, correlations, rankings
4. **Cross-Country Analysis**: Interactive scatter plots with animation
5. **Statistics**: Summary statistics and regression results

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python -m inequality_explorer.src.data_generator

# Run dashboard
python inequality_explorer/app.py

# Open browser to http://localhost:8050
```

## Architecture

```
inequality_explorer/
├── app.py                              # Dash application
├── src/
│   ├── data_generator.py               # Synthetic World Bank data
│   ├── api_ingestion.py                # World Bank API client
│   ├── analysis.py                     # Statistical analysis
│   ├── visualizations_choropleth.py    # Map visualizations
│   └── visualizations_charts.py        # Chart visualizations
├── configs/
│   └── default.yaml                    # Configuration
├── data/                               # Generated/cached data
└── outputs/                            # Exported visualizations
```

## Data Sources

- **Synthetic Data**: Generated with realistic distributions matching World Bank patterns
- **World Bank API**: Optional live data ingestion (requires internet)

## Key Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Gini Coefficient | Income inequality measure | 0-100 |
| HDI | Human Development Index | 0-1 |
| GDP per Capita | Economic output per person | $200-$80,000 |
| Income Quintiles | Share of income by 20% groups | Q1-Q5 |

## Dependencies

- plotly, dash - Interactive visualizations
- pandas, numpy - Data manipulation
- scipy - Statistical analysis
- requests - API access
- pyyaml - Configuration
