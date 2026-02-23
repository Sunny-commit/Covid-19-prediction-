# 🦠 COVID-19 Prediction - Time Series & Forecasting

A **machine learning system** for predicting COVID-19 cases, deaths, and trends using time series analysis and epidemiological models.

## 🎯 Overview

This project covers:
- ✅ Time series forecasting
- ✅ ARIMA/SARIMA models
- ✅ Trend analysis
- ✅ Epidemiological models
- ✅ Region-specific predictions
- ✅ Uncertainty quantification
- ✅ Public health insights

## 📈 COVID-19 Data Analysis

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CovidDataAnalyzer:
    """Analyze COVID-19 data"""
    
    def __init__(self):
        self.data = None
    
    def load_covid_data(self, filepath):
        """Load COVID dataset"""
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')
        
        return self.data
    
    def calculate_daily_metrics(self):
        """Calculate daily cases/deaths"""
        df = self.data.copy()
        
        df['daily_cases'] = df['confirmed'].diff()
        df['daily_deaths'] = df['deaths'].diff()
        df['daily_recovered'] = df['recovered'].diff()
        
        # 7-day rolling average (smoothing)
        df['cases_7day_avg'] = df['daily_cases'].rolling(7).mean()
        df['deaths_7day_avg'] = df['daily_deaths'].rolling(7).mean()
        
        return df
    
    def calculate_growth_rate(self):
        """Calculate growth rates"""
        df = self.data.copy()
        
        # Daily growth rate
        df['case_growth_rate'] = df['confirmed'].pct_change() * 100
        df['death_growth_rate'] = df['deaths'].pct_change() * 100
        
        # 7-day growth average
        df['growth_rate_7day'] = df['case_growth_rate'].rolling(7).mean()
        
        return df
    
    def analyze_regional_trends(self):
        """Compare regions"""
        df = self.data.copy()
        
        if 'region' in df.columns:
            regional_stats = df.groupby('region').agg({
                'confirmed': 'max',
                'deaths': 'max',
                'recovered': 'max'
            })
            
            regional_stats['fatality_rate'] = (
                regional_stats['deaths'] / regional_stats['confirmed'] * 100
            )
            
            return regional_stats
    
    def calculate_reproduction_number(self, window=7):
        """Estimate R-value approximation"""
        df = self.data.copy()
        
        df['cases_shift'] = df['daily_cases'].shift(window)
        df['reproduction_proxy'] = df['daily_cases'] / (df['cases_shift'] + 1)
        
        return df
```

## 🔮 ARIMA Time Series Models

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class CovidTimeSeriesPredictor:
    """Time series forecasting"""
    
    def __init__(self):
        self.arima_model = None
        self.sarima_model = None
    
    def stationarity_test(self, data):
        """Augmented Dickey-Fuller test"""
        result = adfuller(data.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def plot_acf_pacf(self, data):
        """Plot autocorrelation"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        plot_acf(data, lags=40, ax=axes[0])
        plot_pacf(data, lags=40, ax=axes[1])
        
        plt.show()
    
    def fit_arima(self, data, order=(1, 1, 1)):
        """Fit ARIMA model"""
        self.arima_model = ARIMA(data, order=order)
        results = self.arima_model.fit()
        
        return results
    
    def fit_sarima(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        """Fit SARIMA (seasonal ARIMA)"""
        self.sarima_model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = self.sarima_model.fit()
        
        return results
    
    def forecast_cases(self, steps=30):
        """Forecast future cases"""
        if self.sarima_model:
            forecast_result = self.sarima_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            return {
                'forecast': forecast,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            }
    
    def plot_forecast(self, data, forecast_result, title='COVID-19 Cases Forecast'):
        """Visualize forecast"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(data.index, data, label='Historical Data')
        plt.plot(forecast_result['forecast'].index, forecast_result['forecast'], 
                label='Forecast', color='red')
        plt.fill_between(forecast_result['lower_ci'].index,
                        forecast_result['lower_ci'],
                        forecast_result['upper_ci'],
                        alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Number of Cases')
        plt.legend()
        plt.grid(True)
        plt.show()
```

## 🧬 Epidemiological Models

```python
class SIRModel:
    """SIR (Susceptible-Infected-Recovered) model"""
    
    def __init__(self, population, initial_infected=1):
        self.N = population  # Total population
        self.I = initial_infected  # Initial infected
        self.S = population - initial_infected  # Initially susceptible
        self.R = 0  # Initially recovered
    
    def sir_dynamics(self, beta, gamma, days):
        """Simulate SIR model"""
        S, I, R = [self.S], [self.I], [self.R]
        
        for day in range(days):
            # New infections
            new_infections = (beta * S[-1] * I[-1]) / self.N
            # New recoveries
            new_recoveries = gamma * I[-1]
            
            # Update compartments
            S.append(S[-1] - new_infections)
            I.append(I[-1] + new_infections - new_recoveries)
            R.append(R[-1] + new_recoveries)
        
        return {
            'susceptible': S,
            'infected': I,
            'recovered': R
        }
    
    def calculate_basic_reproduction(self, beta, gamma):
        """Calculate R0"""
        R0 = beta / gamma
        
        return R0
    
    def plot_sir_curves(self, S, I, R):
        """Visualize SIR model"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(S, label='Susceptible', color='blue')
        plt.plot(I, label='Infected', color='red')
        plt.plot(R, label='Recovered', color='green')
        
        plt.title('SIR Model Simulation')
        plt.xlabel('Days')
        plt.ylabel('Number of People')
        plt.legend()
        plt.grid(True)
        plt.show()
```

## 📊 Evaluation & Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CovidMetricsEvaluator:
    """Evaluate prediction accuracy"""
    
    @staticmethod
    def calculate_forecast_accuracy(y_actual, y_predicted):
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(y_actual, y_predicted)
        rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
        mape = np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    @staticmethod
    def calculate_peak_prediction_error(actual_peak, predicted_peak, actual_peak_date, predicted_peak_date):
        """Error in peak prediction"""
        return {
            'peak_value_error': abs(actual_peak - predicted_peak),
            'peak_date_error_days': abs((actual_peak_date - predicted_peak_date).days)
        }
    
    @staticmethod
    def calculate_attack_rate(confirmed, population):
        """Overall attack rate"""
        return (confirmed / population) * 100
```

## 💡 Interview Talking Points

**Q: ARIMA vs SARIMA?**
```
Answer:
- ARIMA: AR (autoregressive) + MA (moving average)
- SARIMA: Adds seasonal component
- COVID shows strong seasonality
- SARIMA captures weekly patterns
- AIC/BIC for model selection
```

**Q: R0 significance?**
```
Answer:
- Basic Reproduction Number
- Average infections per case
- R0 > 1: Epidemic grows
- R0 < 1: Epidemic dies out
- Guides intervention policy
```

## 🌟 Portfolio Value

✅ Time series forecasting
✅ ARIMA/SARIMA models
✅ Epidemiological modeling
✅ Public health data
✅ Uncertainty quantification
✅ Trend analysis
✅ Real-world pandemic data

---

**Technologies**: Statsmodels, Pandas, NumPy, Scikit-learn

