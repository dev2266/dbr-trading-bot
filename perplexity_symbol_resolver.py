# perplexity_symbol_resolver.py

import logging
import re
import json
import gzip
import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta
import os
from difflib import get_close_matches
import threading
import time

logger = logging.getLogger(__name__)

class PerplexitySymbolResolver:
    """
    Enhanced Perplexity AI symbol resolver that uses complete Upstox instruments data
    Downloads and caches ALL Indian stocks for comprehensive symbol resolution
    """
    
    def __init__(self):
        self.instruments_data = {}
        self.symbol_mappings = {}
        self.name_to_symbol = {}
        self.last_update = None
        self.cache_file = 'upstox_instruments_cache.json'
        self.lock = threading.RLock()
        
        # Upstox instruments URL (contains ALL stocks)
        self.instruments_url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
        
        # Load cached data or download fresh
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize instruments data from cache or download"""
        try:
            # Try loading from cache first
            if self._load_from_cache():
                logger.info(f"[RESOLVER] Loaded {len(self.symbol_mappings)} instruments from cache")
                return
            
            # If cache failed, download fresh data
            logger.info("[RESOLVER] Cache not available, downloading fresh data...")
            if self._download_instruments():
                self._save_to_cache()
                logger.info(f"[RESOLVER] Downloaded and cached {len(self.symbol_mappings)} instruments")
            else:
                # Fallback to basic symbols if download fails
                self._load_fallback_symbols()
                logger.warning("[RESOLVER] Using fallback symbol list")
                
        except Exception as e:
            logger.error(f"[RESOLVER] Initialization failed: {e}")
            self._load_fallback_symbols()
    
    def _load_from_cache(self) -> bool:
        """Load instruments from cache if recent"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            # Check if cache is recent (less than 24 hours old)
            cache_time = os.path.getmtime(self.cache_file)
            if time.time() - cache_time > 24 * 3600:  # 24 hours
                logger.info("[RESOLVER] Cache is older than 24 hours, will refresh")
                return False
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self.symbol_mappings = cache_data.get('symbol_mappings', {})
            self.name_to_symbol = cache_data.get('name_to_symbol', {})
            self.last_update = cache_data.get('last_update')
            
            return len(self.symbol_mappings) > 0
            
        except Exception as e:
            logger.error(f"[RESOLVER] Cache loading failed: {e}")
            return False
    
    def _download_instruments(self) -> bool:
        """Download complete instruments data from Upstox"""
        try:
            logger.info("[RESOLVER] Downloading Upstox instruments data...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.instruments_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Decompress gzipped content
            decompressed_data = gzip.decompress(response.content)
            instruments = json.loads(decompressed_data.decode('utf-8'))
            
            self._process_instruments(instruments)
            self.last_update = datetime.now().isoformat()
            
            logger.info(f"[RESOLVER] Successfully processed {len(instruments)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"[RESOLVER] Download failed: {e}")
            return False
    
    def _process_instruments(self, instruments: List[Dict]):
        """Process instruments data and create mappings"""
        try:
            self.symbol_mappings.clear()
            self.name_to_symbol.clear()
            
            for instrument in instruments:
                # Focus on equity segments
                segment = instrument.get('segment', '')
                if segment not in ['NSE_EQ', 'BSE_EQ']:
                    continue
                
                instrument_type = instrument.get('instrument_type', '')
                if instrument_type != 'EQ':
                    continue
                
                name = instrument.get('name', '').strip()
                trading_symbol = instrument.get('trading_symbol', '').strip()
                short_name = instrument.get('short_name', '').strip()
                
                if not name or not trading_symbol:
                    continue
                
                # Create multiple mappings for each stock
                self._add_symbol_mapping(name.lower(), trading_symbol)
                self._add_symbol_mapping(trading_symbol.lower(), trading_symbol)
                
                if short_name and short_name != name:
                    self._add_symbol_mapping(short_name.lower(), trading_symbol)
                
                # Create variations
                self._create_name_variations(name, trading_symbol)
                
                # Store in name-to-symbol mapping for fuzzy search
                self.name_to_symbol[name.lower()] = trading_symbol
                if short_name:
                    self.name_to_symbol[short_name.lower()] = trading_symbol
            
            logger.info(f"[RESOLVER] Created {len(self.symbol_mappings)} symbol mappings")
            
        except Exception as e:
            logger.error(f"[RESOLVER] Processing failed: {e}")
    
    def _add_symbol_mapping(self, key: str, symbol: str):
        """Add a symbol mapping with conflict resolution"""
        key = key.strip().lower()
        if key and len(key) >= 2:
            self.symbol_mappings[key] = symbol
    
    def _create_name_variations(self, name: str, symbol: str):
        """Create variations of company names for better matching"""
        try:
            name_lower = name.lower()
            
            # Remove common suffixes
            variations = [name_lower]
            
            # Remove suffixes like "limited", "ltd", "pvt", etc.
            suffixes = [
                'limited', 'ltd', 'ltd.', 'pvt', 'private', 'company', 'corp', 
                'corporation', 'enterprises', 'industries', 'inc', 'co'
            ]
            
            for suffix in suffixes:
                if name_lower.endswith(f' {suffix}'):
                    variation = name_lower.replace(f' {suffix}', '').strip()
                    variations.append(variation)
                elif name_lower.endswith(suffix):
                    variation = name_lower.replace(suffix, '').strip()
                    variations.append(variation)
            
            # Add each variation to mappings
            for variation in set(variations):
                if variation and len(variation) >= 2:
                    self.symbol_mappings[variation] = symbol
                    
        except Exception as e:
            logger.error(f"[RESOLVER] Name variation creation failed: {e}")
    
    def _save_to_cache(self):
        """Save instruments data to cache"""
        try:
            cache_data = {
                'symbol_mappings': self.symbol_mappings,
                'name_to_symbol': self.name_to_symbol,
                'last_update': self.last_update,
                'cache_timestamp': time.time()
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"[RESOLVER] Cached {len(self.symbol_mappings)} mappings")
            
        except Exception as e:
            logger.error(f"[RESOLVER] Cache saving failed: {e}")
    
    def _load_fallback_symbols(self):
        """Load basic fallback symbols if download fails"""
        self.symbol_mappings = {
            'reliance': 'RELIANCE', 'reliance industries': 'RELIANCE', 'ril': 'RELIANCE',
            'tcs': 'TCS', 'tata consultancy services': 'TCS',
            'infosys': 'INFY', 'infy': 'INFY',
            'hdfc bank': 'HDFCBANK', 'hdfcbank': 'HDFCBANK', 'hdfc': 'HDFCBANK',
            'icici bank': 'ICICIBANK', 'icici': 'ICICIBANK',
            'sbi': 'SBIN', 'state bank of india': 'SBIN',
            'bajaj finance': 'BAJFINANCE', 'bajaj': 'BAJFINANCE',
            'asian paints': 'ASIANPAINT', 'maruti': 'MARUTI', 'maruti suzuki': 'MARUTI',
            'tata motors': 'TATAMOTORS', 'wipro': 'WIPRO', 'titan': 'TITAN',
            'adani enterprises': 'ADANIENT', 'britannia': 'BRITANNIA'
        }
        self.name_to_symbol = dict(self.symbol_mappings)
        logger.warning("[RESOLVER] Using limited fallback symbol set")
    
    def resolve_stock_symbol(self, user_input: str) -> Dict[str, Any]:
        """
        Resolve user input to a stock symbol using comprehensive Upstox data
        
        Args:
            user_input (str): User's input text
            
        Returns:
            Dict containing resolution result
        """
        try:
            with self.lock:
                # Clean and normalize input
                cleaned_input = self._clean_input(user_input)
                
                logger.info(f"[RESOLVER] Processing: '{user_input}' -> '{cleaned_input}'")
                
                # Method 1: Direct exact match
                if cleaned_input in self.symbol_mappings:
                    resolved_symbol = self.symbol_mappings[cleaned_input]
                    return self._success_result(resolved_symbol, user_input, 'direct_match', 95)
                
                # Method 2: Partial matching within keys
                for key, symbol in self.symbol_mappings.items():
                    if cleaned_input in key or key in cleaned_input:
                        return self._success_result(symbol, user_input, 'partial_match', 85)
                
                # Method 3: Fuzzy matching using company names
                fuzzy_result = self._fuzzy_match(cleaned_input)
                if fuzzy_result:
                    return fuzzy_result
                
                # Method 4: Check if input looks like a valid symbol
                if self._looks_like_symbol(cleaned_input):
                    potential_symbol = cleaned_input.upper()
                    return self._success_result(potential_symbol, user_input, 'assumed_symbol', 70)
                
                # Method 5: Word-based matching
                word_match = self._word_based_match(cleaned_input)
                if word_match:
                    return word_match
                
                # No match found
                return self._failure_result(user_input, cleaned_input)
                
        except Exception as e:
            logger.error(f"[RESOLVER] Resolution failed: {e}")
            return {
                'success': False,
                'error': f'Symbol resolution failed: {str(e)}',
                'original_input': user_input
            }
    
    def _clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""
        cleaned = user_input.strip().lower()
        
        # Remove common words that don't help with symbol resolution
        cleaned = re.sub(r'\b(stock|share|equity|company|ltd|limited|inc|pvt|private)\b', '', cleaned)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove special characters
        cleaned = re.sub(r'\s+', ' ', cleaned)     # Normalize whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _fuzzy_match(self, cleaned_input: str) -> Dict[str, Any]:
        """Perform fuzzy matching on company names"""
        try:
            # Get close matches from company names
            matches = get_close_matches(cleaned_input, self.name_to_symbol.keys(), n=3, cutoff=0.6)
            
            if matches:
                best_match = matches[0]
                symbol = self.name_to_symbol[best_match]
                
                # Calculate confidence based on similarity
                similarity = len(set(cleaned_input.split()) & set(best_match.split())) / len(set(cleaned_input.split()) | set(best_match.split()))
                confidence = int(70 + similarity * 20)
                
                return self._success_result(symbol, cleaned_input, 'fuzzy_match', confidence)
            
            return None
            
        except Exception as e:
            logger.error(f"[RESOLVER] Fuzzy matching failed: {e}")
            return None
    
    def _word_based_match(self, cleaned_input: str) -> Dict[str, Any]:
        """Match based on individual words"""
        try:
            input_words = set(cleaned_input.split())
            if len(input_words) == 0:
                return None
            
            best_match = None
            best_score = 0
            
            for name, symbol in self.name_to_symbol.items():
                name_words = set(name.split())
                
                # Calculate word overlap
                common_words = input_words & name_words
                if len(common_words) > 0:
                    score = len(common_words) / len(input_words | name_words)
                    if score > best_score and score > 0.3:  # Minimum threshold
                        best_score = score
                        best_match = symbol
            
            if best_match:
                confidence = int(60 + best_score * 25)
                return self._success_result(best_match, cleaned_input, 'word_match', confidence)
            
            return None
            
        except Exception as e:
            logger.error(f"[RESOLVER] Word-based matching failed: {e}")
            return None
    
    def _looks_like_symbol(self, cleaned_input: str) -> bool:
        """Check if input looks like a stock symbol"""
        return (len(cleaned_input) <= 15 and 
                cleaned_input.replace(' ', '').isalnum() and 
                not ' ' in cleaned_input)
    
    def _success_result(self, symbol: str, original_input: str, method: str, confidence: int) -> Dict[str, Any]:
        """Create success result dictionary"""
        return {
            'success': True,
            'resolved_symbol': symbol,
            'confidence': confidence,
            'method': method,
            'original_input': original_input,
            'data_source': 'upstox_complete_instruments',
            'total_instruments': len(self.symbol_mappings)
        }
    
    def _failure_result(self, original_input: str, cleaned_input: str) -> Dict[str, Any]:
        """Create failure result dictionary"""
        suggestions = self._get_suggestions(cleaned_input)
        
        return {
            'success': False,
            'error': f'Could not resolve "{original_input}" to a valid stock symbol',
            'suggestions': suggestions,
            'original_input': original_input,
            'cleaned_input': cleaned_input,
            'total_instruments_available': len(self.symbol_mappings)
        }
    
    def _get_suggestions(self, cleaned_input: str) -> List[str]:
        """Get suggestions for similar symbols"""
        try:
            suggestions = []
            
            # Get fuzzy matches for suggestions
            matches = get_close_matches(cleaned_input, self.name_to_symbol.keys(), n=5, cutoff=0.4)
            
            for match in matches:
                symbol = self.name_to_symbol[match]
                if symbol not in suggestions:
                    suggestions.append(symbol)
            
            # If no fuzzy matches, provide popular symbols
            if not suggestions:
                popular_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BAJFINANCE', 'MARUTI']
                suggestions = popular_symbols[:3]
            
            return suggestions[:5]  # Return max 5 suggestions
            
        except Exception as e:
            logger.error(f"[RESOLVER] Suggestion generation failed: {e}")
            return ['RELIANCE', 'TCS', 'HDFCBANK']
    
    def refresh_data(self) -> bool:
        """Manually refresh instruments data"""
        try:
            logger.info("[RESOLVER] Manual data refresh requested")
            if self._download_instruments():
                self._save_to_cache()
                logger.info(f"[RESOLVER] Data refreshed successfully, {len(self.symbol_mappings)} instruments")
                return True
            return False
        except Exception as e:
            logger.error(f"[RESOLVER] Manual refresh failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics"""
        return {
            'total_symbol_mappings': len(self.symbol_mappings),
            'total_companies': len(self.name_to_symbol),
            'last_update': self.last_update,
            'cache_file_exists': os.path.exists(self.cache_file),
            'data_source': 'upstox_complete_instruments' if len(self.symbol_mappings) > 100 else 'fallback'
        }

# Create global instance
perplexity_resolver = PerplexitySymbolResolver()
