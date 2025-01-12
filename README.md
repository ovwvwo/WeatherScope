# WeatherScope
# WeatherScope: Advanced Weather Analysis and Forecasting Tool

## Overview
WeatherScope is a Python-based weather analysis tool that provides comprehensive weather data analysis, visualization, and forecasting capabilities. The tool fetches historical weather data, performs statistical analysis, creates insightful visualizations, and generates weather forecasts using machine learning.

## Features
- Historical weather data retrieval
- Statistical analysis of weather patterns
- Interactive data visualizations
- Temperature forecasting using machine learning
- Correlation analysis between weather parameters

## Installation

### Prerequisites
- Python 3.8+
- OpenWeatherMap API key

### Required Dependencies
```bash
pip install pandas numpy requests matplotlib seaborn scikit-learn
```

### Quick Start
```python
from weatherscope import WeatherAnalysis

# Initialize analyzer with your API key
analyzer = WeatherAnalysis('YOUR_API_KEY')

# Fetch and analyze data
data = analyzer.fetch_historical_data(days=30)
stats, analysis = analyzer.analyze_data(data)

# Generate visualizations and forecast
analyzer.create_visualizations(data)
forecast, metrics = analyzer.build_forecast_model(data)
```

## API Reference

### Class: WeatherAnalysis

#### Constructor
```python
WeatherAnalysis(api_key: str)
```
- `api_key`: Your OpenWeatherMap API key
- Default city is set to Moscow, RU (can be modified in the constructor)

#### Methods

##### fetch_historical_data
```python
fetch_historical_data(days: int = 30) -> pd.DataFrame
```
Fetches historical weather data for the specified number of days.
- Parameters:
  - `days`: Number of days to fetch (default: 30)
- Returns: DataFrame with weather data

##### analyze_data
```python
analyze_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]
```
Performs statistical analysis on weather data.
- Parameters:
  - `df`: DataFrame containing weather data
- Returns:
  - Tuple containing basic statistics and detailed analysis

##### create_visualizations
```python
create_visualizations(df: pd.DataFrame) -> matplotlib.figure.Figure
```
Generates visualization plots for weather data.
- Parameters:
  - `df`: DataFrame containing weather data
- Returns:
  - Matplotlib figure containing four subplots

##### build_forecast_model
```python
build_forecast_model(df: pd.DataFrame, days_ahead: int = 7) -> Tuple[pd.DataFrame, dict]
```
Builds and trains a forecasting model.
- Parameters:
  - `df`: DataFrame containing weather data
  - `days_ahead`: Number of days to forecast (default: 7)
- Returns:
  - Tuple containing forecast DataFrame and model metrics

## Data Structure
The tool works with the following weather parameters:
- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (m/s)

## Example Usage

```python
# Initialize the analyzer
analyzer = WeatherAnalysis('YOUR_API_KEY')

# Fetch 30 days of historical data
data = analyzer.fetch_historical_data(days=30)

# Perform analysis
stats, analysis = analyzer.analyze_data(data)
print("Temperature trend:", analysis['temperature_trend'])
print("Temperature-Humidity correlation:", analysis['temp_humidity_corr'])

# Create visualizations
fig = analyzer.create_visualizations(data)
plt.show()

# Generate forecast
forecast, metrics = analyzer.build_forecast_model(data)
print("Forecast accuracy (R²):", metrics['r2'])
```

## Visualization Outputs
The tool generates four types of visualizations:
1. Daily Temperature Plot
2. Temperature-Humidity Correlation Scatter Plot
3. Temperature Distribution Histogram
4. Weather Parameters Correlation Heatmap

## Model Metrics
The forecasting model provides the following metrics:
- Mean Squared Error (MSE)
- R-squared (R²) Score

## Error Handling
The tool includes error handling for:
- API connection issues
- Invalid data formats
- Missing values
- Model training errors

## Contributing
To contribute to WeatherScope:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request


## Acknowledgments
- OpenWeatherMap API for providing weather data
- Scientific Python community for the analytical tools
