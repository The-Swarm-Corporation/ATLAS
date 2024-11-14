from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import yfinance as yf
from loguru import logger
import numba
import warnings
import threading
import time
import talib
import queue

warnings.filterwarnings("ignore")

logger.add(
    "risk_analysis_{time}.log",
    rotation="1 day",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
)


class DataFetcher:
    """Handles historical and real-time data fetching"""

    def __init__(self, symbol: str, history_years: int = 10):
        self.symbol = symbol
        self.history_years = history_years
        self.data_queue = queue.Queue()
        self._stop_event = threading.Event()

    def get_historical_data(self) -> pd.DataFrame:
        """Fetch multiple years of historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=365 * self.history_years
            )

            logger.info(
                f"Fetching {self.history_years} years of data for {self.symbol}"
            )

            ticker = yf.Ticker(self.symbol)
            data = ticker.history(
                start=start_date, end=end_date, interval="1d"
            )

            if data.empty:
                raise ValueError(
                    f"No historical data available for {self.symbol}"
                )

            logger.info(
                f"Retrieved {len(data)} days of historical data"
            )
            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def start_real_time_updates(self):
        """Start real-time data collection thread"""

        def update_thread():
            while not self._stop_event.is_set():
                try:
                    ticker = yf.Ticker(self.symbol)
                    latest = ticker.history(
                        period="1d", interval="1m"
                    )

                    if not latest.empty:
                        self.data_queue.put(latest)

                    time.sleep(60)  # Update every minute

                except Exception as e:
                    logger.error(f"Real-time update error: {str(e)}")
                    time.sleep(5)  # Back off on error

        self.update_thread = threading.Thread(target=update_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Started real-time data collection")

    def stop_updates(self):
        """Stop real-time updates"""
        self._stop_event.set()
        if hasattr(self, "update_thread"):
            self.update_thread.join()


class FeatureEngineering:
    """Advanced feature engineering for risk prediction"""

    @staticmethod
    @numba.jit(nopython=True)
    def calculate_volatility_features(
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate volatility-based features"""
        windows = [5, 21, 63, 252]
        n = len(returns)
        vol_features = np.zeros((n, len(windows)))

        for i, window in enumerate(windows):
            for j in range(window, n):
                vol_features[j, i] = np.std(
                    returns[j - window : j]
                ) * np.sqrt(252)

        return vol_features

    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        features = pd.DataFrame(index=df.index)

        # Return-based features
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"]).diff()

        # Volatility features
        vol_features = (
            FeatureEngineering.calculate_volatility_features(
                features["returns"].fillna(0).values
            )
        )
        for i, window in enumerate([5, 21, 63, 252]):
            features[f"volatility_{window}d"] = vol_features[:, i]

        # Technical indicators
        features["rsi"] = talib.RSI(df["Close"].values)
        features["macd"], _, _ = talib.MACD(df["Close"].values)
        features["atr"] = talib.ATR(
            df["High"].values, df["Low"].values, df["Close"].values
        )

        # Volume features
        features["volume_ma"] = df["Volume"].rolling(window=21).mean()
        features["volume_std"] = df["Volume"].rolling(window=21).std()
        features["volume_ratio"] = (
            df["Volume"] / features["volume_ma"]
        )

        # Price features
        for window in [5, 21, 63, 252]:
            features[f"price_ma_{window}d"] = (
                df["Close"].rolling(window=window).mean()
            )
            features[f"price_std_{window}d"] = (
                df["Close"].rolling(window=window).std()
            )

        return features.fillna(0)

class RiskPredictor:
    """Real-time risk prediction system with ensemble modeling and adaptive learning"""

    def __init__(
        self,
        symbol: str,
        history_years: int = 10,
        prediction_window: int = 21,
    ):
        self.symbol = symbol
        self.history_years = history_years
        self.prediction_window = prediction_window

        self.data_fetcher = DataFetcher(symbol, history_years)
        
        # Create an ensemble of models with different architectures
        self.models = {
            'xgb': xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                tree_method="hist",
            ),
            'xgb_deep': xgb.XGBRegressor(
                objective="reg:squarederror", 
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=3,
                tree_method="hist",
            ),
            'xgb_shallow': xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=150, 
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=7,
                tree_method="hist",
            )
        }

        self.model_weights = {model: 1/len(self.models) for model in self.models}
        self.scaler = RobustScaler()
        
        # Performance tracking
        self.model_performance = {model: [] for model in self.models}
        self.prediction_window_performance = []
        
        # For real-time predictions
        self.latest_features = None
        self.prediction_history = []
        self._stop_event = threading.Event()
        
        # Adaptive learning parameters
        self.retrain_threshold = 0.1  # Retrain if error exceeds this
        self.adaptation_window = 100  # Number of predictions to assess performance

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        for model in self.models:
            if len(self.model_performance[model]) > self.adaptation_window:
                recent_perf = np.mean(self.model_performance[model][-self.adaptation_window:])
                self.model_weights[model] = np.exp(-recent_perf)
        
        # Normalize weights
        total = sum(self.model_weights.values())
        self.model_weights = {k: v/total for k, v in self.model_weights.items()}

    def train_historical(self):
        """Train on historical data with ensemble approach"""
        logger.info(f"Starting historical training for {self.symbol}")

        historical_data = self.data_fetcher.get_historical_data()
        features = FeatureEngineering.generate_features(historical_data)

        future_vol = (
            features["returns"]
            .rolling(self.prediction_window)
            .std()
            .shift(-self.prediction_window)
        )

        mask = ~future_vol.isna()
        X = features[mask]
        y = future_vol[mask]

        X_scaled = self.scaler.fit_transform(X)

        # Train each model with time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            scores = []
            logger.info(f"Training {model_name}")
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred)
                scores.append(score)
                self.model_performance[model_name].append(score)

            logger.info(
                f"{model_name} CV R2 scores: {np.mean(scores):.3f} (±{np.std(scores):.3f})"
            )
            
            # Final fit
            model.fit(X_scaled, y)

        self._update_model_weights()
        logger.info(f"Model weights after training: {self.model_weights}")
        self.latest_features = features.iloc[-1]

    def start_real_time_predictions(self):
        """Start real-time prediction thread with ensemble predictions"""
        self.data_fetcher.start_real_time_updates()

        def prediction_thread():
            while not self._stop_event.is_set():
                try:
                    latest_data = self.data_fetcher.data_queue.get(timeout=120)
                    if latest_data.empty:
                        continue

                    latest_features = FeatureEngineering.generate_features(latest_data)
                    X_scaled = self.scaler.transform(latest_features.iloc[-1:])
                    
                    # Get weighted predictions from all models
                    predictions = {}
                    for model_name, model in self.models.items():
                        pred = model.predict(X_scaled)[0]
                        predictions[model_name] = pred * self.model_weights[model_name]
                    
                    # Weighted ensemble prediction
                    final_prediction = sum(predictions.values())

                    self.prediction_history.append({
                        "timestamp": datetime.now(),
                        "prediction": final_prediction,
                        "model_predictions": predictions,
                        "price": latest_data["Close"].iloc[-1],
                        "volume": latest_data["Volume"].iloc[-1],
                    })

                    if len(self.prediction_history) > 1000:
                        self.prediction_history = self.prediction_history[-1000:]

                    logger.info(
                        f"New ensemble prediction for {self.symbol}: "
                        f"volatility={final_prediction:.4f}"
                    )

                except queue.Empty:
                    logger.warning("No new data received for 120 seconds")
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    time.sleep(5)

        self.prediction_thread = threading.Thread(target=prediction_thread)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        logger.info("Started real-time predictions")

    def stop(self):
        """Stop all threads"""
        self._stop_event.set()
        self.data_fetcher.stop_updates()
        if hasattr(self, "prediction_thread"):
            self.prediction_thread.join()

def main():
    """Main function to demonstrate usage"""
    try:
        # Initialize predictor for AAPL with 10 years of history
        predictor = RiskPredictor("AAPL", history_years=10)

        # Train on historical data
        predictor.train_historical()

        # Start real-time predictions
        predictor.start_real_time_predictions()

        # Keep running and periodically show latest predictions
        try:
            while True:
                if predictor.prediction_history:
                    latest = predictor.prediction_history[-1]
                    print(
                        f"\nLatest prediction ({latest['timestamp']}):"
                    )
                    print(
                        f"Predicted volatility: {latest['prediction']:.4f}"
                    )
                    print(f"Current price: ${latest['price']:.2f}")
                time.sleep(60)

        except KeyboardInterrupt:
            print("\nStopping predictor...")
            predictor.stop()

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
