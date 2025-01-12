import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class WeatherAnalysis:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.city = "Moscow"
        self.country_code = "RU"
        
    def generate_test_data(self, days=30):
        """Генерация тестовых данных для демонстрации"""
        np.random.seed(42)
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days)]
        
        # Генерируем реалистичные данные для Москвы
        data = {
            'date': dates,
            'temperature': np.random.normal(20, 5, days),  # Средняя температура 20°C
            'humidity': np.random.normal(70, 10, days),    # Средняя влажность 70%
            'pressure': np.random.normal(1013, 5, days),   # Среднее давление 1013 hPa
            'wind_speed': np.random.normal(5, 2, days)     # Средняя скорость ветра 5 м/с
        }
        
        return pd.DataFrame(data)
        
    def fetch_historical_data(self, days=30):
        """Получение исторических данных о погоде"""
        if self.api_key is None:
            print("API ключ не предоставлен. Используем тестовые данные.")
            return self.generate_test_data(days)
            
        weather_data = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            params = {
                'q': f"{self.city},{self.country_code}",
                'appid': self.api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if response.status_code == 200:
                    weather_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'wind_speed': data['wind']['speed']
                    })
            except Exception as e:
                print(f"Ошибка получения данных для {date}: {e}")
        
        if not weather_data:
            print("Не удалось получить данные через API. Используем тестовые данные.")
            return self.generate_test_data(days)
            
        return pd.DataFrame(weather_data)
    
    def analyze_data(self, df):
        """Анализ погодных данных"""
        if df.empty:
            raise ValueError("DataFrame пустой. Проверьте получение данных.")
            
        # Проверка наличия необходимых колонок
        required_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
        
        # Базовая статистика
        stats = df.describe()
        
        # Расчет дополнительных метрик
        analysis = {
            'temperature_trend': np.polyfit(range(len(df)), df['temperature'], 1)[0],
            'temp_humidity_corr': df['temperature'].corr(df['humidity']),
            'temp_pressure_corr': df['temperature'].corr(df['pressure']),
            'extreme_temps': {
                'max': df['temperature'].max(),
                'min': df['temperature'].min(),
                'max_date': df.loc[df['temperature'].idxmax(), 'date'],
                'min_date': df.loc[df['temperature'].idxmin(), 'date']
            }
        }
        
        return stats, analysis
    
    def create_visualizations(self, df):
        """Создание визуализаций"""
        if df.empty:
            raise ValueError("DataFrame пустой. Проверьте получение данных.")
            
        # Настройка стиля
        plt.style.use('seaborn')
        
        # График температуры
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Температура по дням
        axes[0, 0].plot(df['date'], df['temperature'], marker='o')
        axes[0, 0].set_title('Температура по дням')
        axes[0, 0].set_xlabel('Дата')
        axes[0, 0].set_ylabel('Температура (°C)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Корреляция температуры и влажности
        axes[0, 1].scatter(df['temperature'], df['humidity'])
        axes[0, 1].set_title('Корреляция температуры и влажности')
        axes[0, 1].set_xlabel('Температура (°C)')
        axes[0, 1].set_ylabel('Влажность (%)')
        
        # Распределение температуры
        sns.histplot(data=df, x='temperature', bins=15, ax=axes[1, 0])
        axes[1, 0].set_title('Распределение температуры')
        axes[1, 0].set_xlabel('Температура (°C)')
        
        # Тепловая карта корреляций
        correlation_matrix = df[['temperature', 'humidity', 'pressure', 'wind_speed']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Тепловая карта корреляций')
        
        plt.tight_layout()
        return fig
    
    def build_forecast_model(self, df, days_ahead=7):
        """Построение прогнозной модели"""
        if df.empty:
            raise ValueError("DataFrame пустой. Проверьте получение данных.")
            
        # Подготовка данных
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['temperature'].values
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Прогноз на будущее
        future_dates = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
        future_temps = model.predict(future_dates)
        
        forecast = pd.DataFrame({
            'date': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_ahead + 1)],
            'predicted_temperature': future_temps
        })
        
        return forecast, {'mse': mse, 'r2': r2}

# Пример использования
if __name__ == "__main__":
    # Создаем анализатор без API ключа для использования тестовых данных
    analyzer = WeatherAnalysis()
    
    try:
        # Получение данных
        data = analyzer.fetch_historical_data(days=30)
        
        # Анализ данных
        stats, analysis = analyzer.analyze_data(data)
        print("\nСтатистика:")
        print(stats)
        print("\nАнализ:")
        print(analysis)
        
        # Создание визуализаций
        fig = analyzer.create_visualizations(data)
        plt.show()
        
        # Построение прогноза
        forecast, metrics = analyzer.build_forecast_model(data)
        print("\nПрогноз на следующие 7 дней:")
        print(forecast)
        print("\nМетрики модели:")
        print(metrics)
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
