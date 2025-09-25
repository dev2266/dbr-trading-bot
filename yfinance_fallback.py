#!/usr/bin/env python3
"""
Enhanced yfinance fallback system for Indian stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedYFinanceFetcher:
    """Enhanced yfinance with retry logic and fallbacks"""
    
    def __init__(self):
        self.session = None
        self.timeout = 15
        self.max_retries = 3
        
    def get_stock_data_with_fallbacks(self, symbol: str, interval: str = '1h') -> Optional[Dict[str, Any]]:
        """Get stock data with multiple fallback strategies"""
        
        # Try different symbol formats
        symbol_formats = [
            f"{symbol}.NS",  # NSE
            f"{symbol}.BO",  # BSE
            symbol,          # Raw symbol
        ]
        
        for symbol_format in symbol_formats:
            try:
                result = self._fetch_with_retry(symbol_format, interval)
                if result:
                    logger.info(f"✅ Success with format: {symbol_format}")
                    return result
            except Exception as e:
                logger.warning(f"Failed format {symbol_format}: {e}")
                continue
                
        return None
    
    def _fetch_with_retry(self, ticker_symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Fetch data with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 * attempt)  # Progressive delay
                
                ticker = yf.Ticker(ticker_symbol)
                
                # Try different period/interval combinations
                combinations = [
                    ('7d', '5m'), ('30d', '15m'), ('60d', '1h'), 
                    ('1y', '1d'), ('2y', '1wk')
                ]
                
                for period, yf_interval in combinations:
                    try:
                        hist = ticker.history(
                            period=period,
                            interval=yf_interval,
                            timeout=self.timeout,
                            auto_adjust=True
                        )
                        
                        if hist is not None and not hist.empty and len(hist) >= 20:
                            return self._process_historical_data(hist, ticker_symbol)
                            
                    except Exception as combo_error:
                        logger.debug(f"Combo {period}/{yf_interval} failed: {combo_error}")
                        continue
                        
            except Exception as attempt_error:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker_symbol}: {attempt_error}")
                if attempt == self.max_retries - 1:
                    raise
                    
        return None
    
    def _process_historical_data(self, hist: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Process historical data into analysis format"""
        try:
            current_price = float(hist['Close'].iloc[-1])
            
            # Basic technical analysis
            close = hist['Close']
            
            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50.0
            
            # Moving averages
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else current_price
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else current_price
            
            # ATR for risk management
            high = hist['High']
            low = hist['Low']
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else current_price * 0.02
            
            # Signal generation
            signal = 'NEUTRAL'
            if current_rsi < 30 and current_price > sma_20:
                signal = 'BUY'
            elif current_rsi > 70 and current_price < sma_20:
                signal = 'SELL'
            elif current_price > sma_50 * 1.05:
                signal = 'BUY'
            elif current_price < sma_50 * 0.95:
                signal = 'SELL'
            
            return {
                'symbol': symbol.replace('.NS', '').replace('.BO', ''),
                'current_price': round(current_price, 2),
                'rsi': round(current_rsi, 2),
                'sma_20': round(float(sma_20), 2),
                'sma_50': round(float(sma_50), 2),
                'atr': round(float(atr), 2),
                'signal': signal,
                'confidence': 75,
                'data_quality': 'ENHANCED_FALLBACK',
                'candles_used': len(hist),
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            }
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            return None

# Global instance
enhanced_yfinance = EnhancedYFinanceFetcher()
