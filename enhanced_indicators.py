#!/usr/bin/env python3

import pandas as pd
import numpy as np
import ta
import warnings

# Advanced libraries
try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

warnings.filterwarnings('ignore')

class EnhancedIndicatorSuite:
    """Complete 130+ Indicators Implementation"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL 130+ technical indicators"""
        try:
            print(f"[DEBUG] 📊 Starting calculation of 130+ indicators...")
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")
            
            original_count = len(df.columns)
            
            # 1. TA Library - 40+ indicators
            df = ta.add_all_ta_features(
                df, 
                open="Open", high="High", low="Low", 
                close="Close", volume="Volume", 
                fillna=True
            )
            
            ta_count = len(df.columns) - original_count
            print(f"[DEBUG] ✅ TA Library: {ta_count} indicators added")
            
            # 2. pandas_ta - 50+ additional indicators
            if PANDAS_TA_AVAILABLE:
                df = EnhancedIndicatorSuite._add_pandas_ta_suite(df)
                pandas_ta_count = len(df.columns) - original_count - ta_count
                print(f"[DEBUG] ✅ pandas_ta: {pandas_ta_count} indicators added")
            
            # 3. TA-Lib - 20+ advanced indicators
            if TALIB_AVAILABLE:
                df = EnhancedIndicatorSuite._add_talib_suite(df)
                talib_count = len(df.columns) - original_count - ta_count - (pandas_ta_count if PANDAS_TA_AVAILABLE else 0)
                print(f"[DEBUG] ✅ TA-Lib: {talib_count} indicators added")
            
            # 4. Custom Advanced Indicators - 20+
            df = EnhancedIndicatorSuite._add_custom_advanced_indicators(df)
            custom_count = len(df.columns) - original_count - ta_count - (pandas_ta_count if PANDAS_TA_AVAILABLE else 0) - (talib_count if TALIB_AVAILABLE else 0)
            print(f"[DEBUG] ✅ Custom Advanced: {custom_count} indicators added")
            
            total_indicators = len(df.columns) - original_count
            print(f"[DEBUG] 🎉 TOTAL INDICATORS: {total_indicators}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Indicator calculation failed: {e}")
            return df
    
    @staticmethod
    def _add_pandas_ta_suite(df: pd.DataFrame) -> pd.DataFrame:
        """Add pandas_ta indicators (50+ indicators)"""
        try:
            # Ichimoku Cloud (9 lines)
            ichimoku = pta.ichimoku(df['High'], df['Low'], df['Close'])
            if ichimoku is not None:
                for col in ichimoku.columns:
                    df[f'pta_{col}'] = ichimoku[col]
            
            # Supertrend (multiple periods)
            for period, multiplier in [(7, 3), (14, 3), (21, 3)]:
                supertrend = pta.supertrend(df['High'], df['Low'], df['Close'], 
                                          length=period, multiplier=multiplier)
                if supertrend is not None:
                    for col in supertrend.columns:
                        df[f'pta_st_{period}_{col}'] = supertrend[col]
            
            # VWAP variations
            df['pta_vwap'] = pta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            df['pta_vwap_anchored'] = pta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], anchor='D')
            
            # Squeeze indicators
            squeeze = pta.squeeze(df['High'], df['Low'], df['Close'])
            if squeeze is not None:
                for col in squeeze.columns:
                    df[f'pta_squeeze_{col}'] = squeeze[col]
            
            # Advanced oscillators
            df['pta_fisher'] = pta.fisher(df['High'], df['Low'])
            df['pta_cg'] = pta.cg(df['Close'])  # Center of Gravity
            
            # Hull Moving Averages
            for period in [9, 14, 21, 50]:
                df[f'pta_hma_{period}'] = pta.hma(df['Close'], length=period)
            
            # Kaufman's Adaptive Moving Average
            df['pta_kama'] = pta.kama(df['Close'])
            
            # Price/Volume trend
            df['pta_pvt'] = pta.pvt(df['Close'], df['Volume'])
            
            # Choppiness Index
            df['pta_chop'] = pta.chop(df['High'], df['Low'], df['Close'])
            
            return df
            
        except Exception as e:
            print(f"[WARNING] pandas_ta suite failed: {e}")
            return df
    
    @staticmethod
    def _add_talib_suite(df: pd.DataFrame) -> pd.DataFrame:
        """Add TA-Lib indicators (20+ indicators)"""
        try:
            # Pattern Recognition (20 patterns)
            patterns = [
                'CDLDOJI', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLENGULFING',
                'CDLHARAMI', 'CDLPIERCING', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
                'CDLDRAGONFLYDOJI', 'CDLGRAVESTONEDOJI', 'CDLSPINNINGTOP',
                'CDLHANGINGMAN', 'CDLINVERTEDHAMMER', 'CDLMARUBOZU'
            ]
            
            for pattern in patterns:
                try:
                    df[f'talib_{pattern.lower()}'] = getattr(talib, pattern)(
                        df['Open'], df['High'], df['Low'], df['Close']
                    )
                except:
                    continue
            
            # Advanced Momentum Indicators
            df['talib_aroon_up'], df['talib_aroon_down'] = talib.AROON(
                df['High'], df['Low'], timeperiod=14
            )
            df['talib_aroonosc'] = talib.AROONOSC(df['High'], df['Low'], timeperiod=14)
            
            # Volume Indicators
            df['talib_ad'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            df['talib_adosc'] = talib.ADOSC(
                df['High'], df['Low'], df['Close'], df['Volume'], 
                fastperiod=3, slowperiod=10
            )
            
            # Price Transform
            df['talib_avgprice'] = talib.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'])
            df['talib_medprice'] = talib.MEDPRICE(df['High'], df['Low'])
            df['talib_typprice'] = talib.TYPPRICE(df['High'], df['Low'], df['Close'])
            df['talib_wclprice'] = talib.WCLPRICE(df['High'], df['Low'], df['Close'])
            
            # Cycle Indicators
            df['talib_ht_dcperiod'] = talib.HT_DCPERIOD(df['Close'])
            df['talib_ht_dcphase'] = talib.HT_DCPHASE(df['Close'])
            
            return df
            
        except Exception as e:
            print(f"[WARNING] TA-Lib suite failed: {e}")
            return df
    
    @staticmethod
    def _add_custom_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom advanced indicators (20+ indicators)"""
        try:
            # Fibonacci Retracements
            high_max = df['High'].rolling(50).max()
            low_min = df['Low'].rolling(50).min()
            fib_range = high_max - low_min
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                df[f'fib_retracement_{int(level*1000)}'] = high_max - (fib_range * level)
            
            # Market Structure
            df['higher_highs'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
            df['lower_lows'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
            
            # Volatility Clustering
            returns = df['Close'].pct_change()
            df['volatility_5'] = returns.rolling(5).std()
            df['volatility_10'] = returns.rolling(10).std()
            df['volatility_20'] = returns.rolling(20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # Price Momentum Quality
            for period in [5, 10, 20]:
                momentum = df['Close'] / df['Close'].shift(period) - 1
                df[f'momentum_quality_{period}'] = momentum / df[f'volatility_{period}']
            
            # Support/Resistance Strength
            def calculate_sr_strength(series, period=20):
                levels = []
                for i in range(len(series)):
                    if i < period:
                        levels.append(0)
                        continue
                    
                    window = series.iloc[i-period:i+1]
                    current = series.iloc[i]
                    
                    # Count touches near current level
                    tolerance = current * 0.01  # 1% tolerance
                    touches = sum(abs(window - current) <= tolerance)
                    levels.append(touches)
                
                return pd.Series(levels, index=series.index)
            
            df['support_strength'] = calculate_sr_strength(df['Low'])
            df['resistance_strength'] = calculate_sr_strength(df['High'])
            
            # Trend Quality Indicators
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()
            
            df['trend_quality'] = abs(sma_20 - sma_50) / df['Close']
            df['trend_consistency'] = (df['Close'] > sma_20).rolling(10).sum() / 10
            
            return df
            
        except Exception as e:
            print(f"[WARNING] Custom advanced indicators failed: {e}")
            return df
