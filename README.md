# 🦠 COVID-19 Prediction - Epidemiological Forecasting

A **machine learning model for COVID-19 case prediction** using time-series analysis and epidemiological features to forecast infection trends, helping healthcare systems prepare resources and implement preventive measures.

## 🎯 Overview

This project provides:
- ✅ Time-series forecasting (LSTM, ARIMA)
- ✅ Epidemiological modeling (SIR, SEIR)
- ✅ Feature engineering from medical data
- ✅ Multi-step ahead predictions
- ✅ Uncertainty quantification
- ✅ Validation against real-world data

## 📊 Data Features

```
Temporal Features:
├── Daily confirmed cases
├── Daily deaths
├── Daily recoveries
├── Test positive rate
└── Hospitalization rate

Geographic Features:
├── Population density
├── Healthcare capacity
├── Mobility patterns
└── Climate/Weather data

Behavioral Features:
├── Vaccination rate
├── Mask compliance
├── Social distancing index
└── Travel patterns
```

## 🔬 Epidemiological Models

### SIR Model (Susceptible-Infected-Recovered)

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SIRModel:
    """Basic SIR epidemiological model"""
    
    def __init__(self, population, beta, gamma):
        """
        Parameters:
        - population: Total population
        - beta: Transmission rate
        - gamma: Recovery rate
        """
        self.N = population
        self.beta = beta
        self.gamma = gamma
    
    def differential(self, y, t):
        """SIR differential equations"""
        S, I, R = y
        
        dS_dt = -self.beta * S * I / self.N
        dI_dt = self.beta * S * I / self.N - self.gamma * I
        dR_dt = self.gamma * I
        
        return [dS_dt, dI_dt, dR_dt]
    
    def simulate(self, initial_infections, days):
        """Simulate disease progression"""
        S0 = self.N - initial_infections
        I0 = initial_infections
        R0 = 0
        
        y0 = [S0, I0, R0]
        t = np.linspace(0, days, days)
        
        solution = odeint(self.differential, y0, t)
        
        return {
            'Susceptible': solution[:, 0],
            'Infected': solution[:, 1],
            'Recovered': solution[:, 2],
            'time': t
        }

# Example
sir = SIRModel(population=1_000_000, beta=0.3, gamma=0.1)
results = sir.simulate(initial_infections=100, days=365)
```

### SEIR Model (Susceptible-Exposed-Infected-Recovered)

```python
class SEIRModel:
    """SEIR model with incubation period"""
    
    def __init__(self, population, beta, sigma, gamma):
        """
        sigma: Rate of progression from exposed to infected
        """
        self.N = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
    
    def differential(self, y, t):
        """SEIR differential equations"""
        S, E, I, R = y
        
        dS_dt = -self.beta * S * I / self.N
        dE_dt = self.beta * S * I / self.N - self.sigma * E
        dI_dt = self.sigma * E - self.gamma * I
        dR_dt = self.gamma * I
        
        return [dS_dt, dE_dt, dI_dt, dR_dt]
```

## 🤖 ML Forecasting Models

### LSTM for Time-Series

```python
import tensorflow as tf
from tensorflow import keras

class LSTMForecast:
    """LSTM model for COVID-19 case forecasting"""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential([
            keras.layers.LSTM(
                64,
                activation='relu',
                input_shape=(self.sequence_length, 1),
                return_sequences=True
            ),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(7)  # 7-day forecast
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data):
        """Prepare sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - 7):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length:i+self.sequence_length+7])
        
        return np.array(X), np.array(y)
    
    def forecast(self, historical_data, steps=7):
        """Forecast next N days"""
        sequence = historical_data[-self.sequence_length:].reshape(1, -1, 1)
        forecasts = []
        
        for _ in range(steps):
            pred = self.model.predict(sequence, verbose=0)[0]
            forecasts.append(pred[0])
            
            # Update sequence
            sequence = np.append(sequence[0][1:], pred[0])
            sequence = sequence.reshape(1, -1, 1)
        
        return np.array(forecasts)
```

### ARIMA for Time-Series

```python
from statsmodels.tsa.arima.model import ARIMA

class ARIMAForecast:
    """ARIMA model for COVID-19 predictions"""
    
    def __init__(self, order=(5, 1, 2)):
        """
        order = (p, d, q)
        p: AR order
        d: Differencing
        q: MA order
        """
        self.order = order
        self.model = None
    
    def fit(self, data):
        """Fit ARIMA model to historical data"""
        self.model = ARIMA(data, order=self.order)
        self.results = self.model.fit()
    
    def forecast(self, steps=7):
        """Forecast next N steps"""
        forecast = self.results.get_forecast(steps=steps)
        return forecast.conf_int()
    
    @staticmethod
    def find_best_order(data, p_range=range(6), d_range=range(3), q_range=range(3)):
        """Grid search for best ARIMA parameters"""
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        pass
        
        return best_order
```

## 📈 Feature Engineering

```python
class CovidFeatureEngineering:
    """Create features for COVID-19 prediction"""
    
    @staticmethod
    def create_temporal_features(data):
        """Extract temporal patterns"""
        features = {
            'day_of_week': data.index.dayofweek,
            'month': data.index.month,
            'is_weekend': data.index.dayofweek >= 5
        }
        return pd.DataFrame(features, index=data.index)
    
    @staticmethod
    def create_rolling_features(cases, windows=[7, 14, 30]):
        """Create rolling statistics"""
        features = pd.DataFrame(index=cases.index)
        
        for window in windows:
            features[f'rolling_mean_{window}'] = cases.rolling(window).mean()
            features[f'rolling_std_{window}'] = cases.rolling(window).std()
            features[f'rolling_max_{window}'] = cases.rolling(window).max()
        
        return features
    
    @staticmethod
    def create_growth_features(cases):
        """Create growth rate features"""
        features = pd.DataFrame(index=cases.index)
        
        # Day-to-day change
        features['daily_change'] = cases.diff()
        
        # Growth rate
        features['growth_rate'] = cases.pct_change()
        
        # 7-day growth
        features['weekly_growth'] = (
            cases / cases.shift(7) - 1
        ) * 100
        
        return features
```

## 📊 Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CovidEvaluator:
    """Evaluate forecast accuracy"""
    
    @staticmethod
    def evaluate_forecast(actual, predicted):
        """Calculate error metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape  # Mean Absolute Percentage Error
        }
    
    @staticmethod
    def plot_forecast(actual, predicted, confidence_interval=None):
        """Visualize forecast vs actual"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(actual, label='Actual Cases', color='blue', linewidth=2)
        plt.plot(predicted, label='Predicted Cases', color='red', linewidth=2)
        
        if confidence_interval is not None:
            plt.fill_between(
                range(len(confidence_interval)),
                confidence_interval[:, 0],
                confidence_interval[:, 1],
                alpha=0.2,
                color='red'
            )
        
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Cases')
        plt.title('COVID-19 Case Forecast')
        plt.show()
```

## 🏥 Healthcare Resource Planning

```python
class ResourcePlanning:
    """Plan healthcare resources based on forecasts"""
    
    @staticmethod
    def estimate_hospital_beds(forecast_cases, hospitalization_rate=0.05):
        """Estimate bed requirements"""
        return forecast_cases * hospitalization_rate
    
    @staticmethod
    def estimate_icu_beds(forecast_cases, icu_rate=0.02):
        """Estimate ICU bed requirements"""
        return forecast_cases * icu_rate
    
    @staticmethod
    def estimate_ventilators(forecast_cases, ventilator_rate=0.01):
        """Estimate ventilator needs"""
        return forecast_cases * ventilator_rate
    
    @staticmethod
    def staffing_requirements(hospital_beds, staff_per_bed=0.5):
        """Calculate staff requirements"""
        return int(hospital_beds * staff_per_bed)
```

## 💡 Interview Talking Points

**Q: How do SIR and LSTM models differ?**
```
Answer:
- SIR: Mechanistic, based on disease biology
- LSTM: Data-driven, learns from patterns
- Hybrid: Combine both for better results
```

**Q: How handle uncertainty in predictions?**
```
Answer:
- Confidence intervals (ARIMA)
- Ensemble models
- Bayesian methods
- Monte Carlo simulation
```

## 🌟 Portfolio Value

✅ Time-series forecasting
✅ Epidemiological modeling
✅ Deep learning (LSTM)
✅ Statistical models (ARIMA)
✅ Public health applications
✅ Uncertainty quantification

## 📄 License

MIT License - Educational Use

---

**Technologies**: TensorFlow, Statsmodels, SciPy, NumPy

