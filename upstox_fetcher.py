#!/usr/bin/env python3
"""
Fixed Upstox API Integration Module
===================================
Real-time data fetching from Upstox API v2
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import json
import logging

logger = logging.getLogger(__name__)

class UpstoxDataFetcher:
    """Fixed Upstox API data fetcher with correct v2 endpoints"""
    
    def __init__(self, api_key, api_secret, access_token):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v2"
        
        # Fixed headers for API v2
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Api-Version': '2.0',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        print(f"[DEBUG] 🔧 UpstoxDataFetcher initialized with API v2")

    def validate_credentials(self) -> bool:
        """Validate Upstox API credentials"""
        try:
            if not self.access_token:
                print(f"[DEBUG] ❌ No access token provided")
                return False

            print(f"[DEBUG] 🔑 Validating Upstox credentials...")
            response = requests.get(
                f"{self.base_url}/user/profile",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    user_name = data.get('data', {}).get('user_name', 'Unknown')
                    print(f"[DEBUG] ✅ Upstox credentials validated for user: {user_name}")
                    return True
                else:
                    print(f"[DEBUG] ❌ API response error: {data}")
                    return False
            else:
                print(f"[DEBUG] ❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[DEBUG] ❌ Timeout - Upstox server slow")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[DEBUG] ❌ Connection error - Check internet connection")
            return False
        except Exception as e:
            print(f"[DEBUG] ❌ Validation error: {e}")
            return False

    def get_instrument_key(self, symbol: str) -> str:
        """Get correct Upstox instrument key for symbol"""
        from symbol_mapper import symbol_mapper
        
        # Get the full Upstox symbol with ISIN
        upstox_symbol = symbol_mapper.get_upstox_symbol(symbol)
        print(f"[DEBUG] 🔍 Symbol mapping: {symbol} -> {upstox_symbol}")
        return upstox_symbol

    def get_live_quote(self, symbol: str):
        """Get live quote with FIXED parsing logic"""
        try:
            instrument_key = self.get_instrument_key(symbol)
            print(f"[DEBUG] 📡 Fetching live quote: {symbol} -> {instrument_key}")

            url = f"{self.base_url}/market-quote/quotes"
            params = {'instrument_key': instrument_key}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            print(f"[DEBUG] 📡 API Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data'):
                    
                    # FIXED: Robust parsing that handles different key formats
                    quote_data = None
                    
                    # Try to find data using any of the keys in response
                    for key, value in data['data'].items():
                        # Match by symbol name or instrument key
                        if (symbol.upper() in key.upper() or 
                            key.replace(':', '|') == instrument_key or
                            key == instrument_key):
                            quote_data = value
                            print(f"[DEBUG] ✅ Found data with key: {key}")
                            break
                    
                    if quote_data:
                        price = quote_data.get('last_price', 0)
                        print(f"[DEBUG] ✅ Live quote: {symbol} = ₹{price}")
                        
                        return {
                            'last_price': price,
                            'open': quote_data.get('ohlc', {}).get('open', 0),
                            'high': quote_data.get('ohlc', {}).get('high', 0),
                            'low': quote_data.get('ohlc', {}).get('low', 0),
                            'close': quote_data.get('ohlc', {}).get('close', 0),
                            'volume': quote_data.get('volume', 0),
                            'change': quote_data.get('net_change', 0),
                            'change_percent': quote_data.get('percent_change', 0),
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'data_source': 'upstox'
                        }
                
                print(f"[DEBUG] ❌ No matching data found in response")
                return {'error': f'No data found for {symbol}', 'symbol': symbol}
            
            else:
                print(f"[DEBUG] ❌ HTTP error {response.status_code}: {response.text}")
                return {'error': f'HTTP {response.status_code}', 'symbol': symbol}
                
        except Exception as e:
            print(f"[DEBUG] ❌ Live quote error: {e}")
            return {'error': f'Quote error: {str(e)}', 'symbol': symbol}

    def get_historical_data(self, symbol: str, interval: str = '1minute', days: int = 30):
        """Get historical data with FIXED interval mapping"""
        try:
            instrument_key = self.get_instrument_key(symbol)
            print(f"[DEBUG] 📊 Fetching historical data: {symbol} -> {instrument_key}")

            # FIXED: Correct Upstox API v2 interval mapping
            upstox_interval_map = {
                '1m': '1minute',
                '5m': '1minute',      # Use 1minute and downsample
                '15m': '1minute',     # Use 1minute and downsample  
                '30m': '30minute',
                '1h': '30minute',     # Use 30minute and downsample
                '4h': 'day',          # Use daily and downsample
                '1d': 'day',
                '1w': 'week',
                '1M': 'month'
            }
            
            upstox_interval = upstox_interval_map.get(interval, 'day')
            print(f"[DEBUG] 📊 Interval mapping: {interval} -> {upstox_interval}")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # FIXED: Use correct interval format
            url = f"{self.base_url}/historical-candle/{instrument_key}/{upstox_interval}/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"
            
            response = requests.get(url, headers=self.headers, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data', {}).get('candles'):
                    candles = data['data']['candles']
                    if not candles:
                        print(f"[DEBUG] ❌ No historical data available for {symbol}")
                        return None

                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
                    df = df.sort_index()
                    
                    # Downsample if needed (e.g., 1minute -> 1hour)
                    if interval in ['5m', '15m', '1h', '4h'] and upstox_interval in ['1minute', '30minute', 'day']:
                        df = self._resample_data(df, interval)
                    
                    print(f"[DEBUG] ✅ Historical data: {len(df)} candles for {symbol}")
                    return df
                else:
                    print(f"[DEBUG] ❌ No historical data in response: {data}")
                    return None
            else:
                print(f"[DEBUG] ❌ Historical data HTTP {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"[DEBUG] ❌ Historical data error: {e}")
            return None

    def _resample_data(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample data to target interval"""
        try:
            # Mapping for pandas resample frequency
            resample_map = {
                '5m': '5T',
                '15m': '15T', 
                '1h': '1H',
                '4h': '4H'
            }
            
            freq = resample_map.get(target_interval, '1H')
            
            # Resample OHLCV data
            resampled = df.resample(freq).agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            print(f"[DEBUG] 📊 Resampled from {len(df)} to {len(resampled)} candles")
            return resampled
            
        except Exception as e:
            print(f"[DEBUG] ❌ Resampling failed: {e}")
            return df

    def get_market_status(self) -> dict:
        """Get market status from Upstox"""
        try:
            url = f"{self.base_url}/market-quote/market-status/NSE"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                return {'error': f'Market status HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"[DEBUG] ❌ Market status error: {e}")
            return {'error': str(e)}

    def test_connection(self) -> dict:
        """Test Upstox API connection with detailed diagnostics"""
        print(f"\n🧪 UPSTOX API CONNECTION TEST")
        print(f"=" * 40)
        
        results = {
            'credentials_valid': False,
            'market_status_ok': False,
            'sample_quote_ok': False,
            'errors': []
        }
        
        try:
            # Test 1: Credentials
            print(f"1. 🔑 Testing credentials...")
            if self.validate_credentials():
                results['credentials_valid'] = True
                print(f"   ✅ Credentials valid")
            else:
                results['errors'].append("Invalid credentials")
                print(f"   ❌ Credentials invalid")
            
            # Test 2: Market status
            print(f"2. 📊 Testing market status...")
            market_status = self.get_market_status()
            if 'error' not in market_status:
                results['market_status_ok'] = True
                print(f"   ✅ Market status OK")
            else:
                results['errors'].append(f"Market status error: {market_status['error']}")
                print(f"   ❌ Market status failed")
            
            # Test 3: Sample quote
            print(f"3. 💹 Testing sample quote (RELIANCE)...")
            quote = self.get_live_quote('RELIANCE')
            if 'error' not in quote and quote.get('last_price', 0) > 0:
                results['sample_quote_ok'] = True
                print(f"   ✅ Sample quote: RELIANCE = ₹{quote['last_price']}")
            else:
                results['errors'].append(f"Sample quote error: {quote.get('error', 'Unknown')}")
                print(f"   ❌ Sample quote failed: {quote.get('error', 'Unknown')}")
                
        except Exception as e:
            results['errors'].append(f"Test error: {str(e)}")
            print(f"❌ Test failed: {e}")
        
        # Summary
        success_count = sum([results['credentials_valid'], results['market_status_ok'], results['sample_quote_ok']])
        print(f"\n📊 TEST RESULTS: {success_count}/3 tests passed")
        
        if success_count == 3:
            print(f"🎉 ALL TESTS PASSED! Upstox integration working perfectly.")
        else:
            print(f"⚠️ Some tests failed. Check errors above.")
        
        return results
