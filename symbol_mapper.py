#!/usr/bin/env python3
"""
Centralized Symbol Mapping for Trading Bot
==========================================
Complete solution for Indian stock symbol mapping
"""

class SymbolMapper:
    """Centralized symbol mapping for Indian stocks"""
    
    # Complete symbol mapping with ISIN codes for Upstox
    SYMBOL_MAP = {
        # Banking Stocks
        'HDFCBANK': {
            'yahoo': 'HDFCBANK.NS',
            'upstox': 'NSE_EQ|INE040A01034',
            'name': 'HDFC Bank Limited',
            'sector': 'Banking'
        },
        'ICICIBANK': {
            'yahoo': 'ICICIBANK.NS',
            'upstox': 'NSE_EQ|INE090A01021',
            'name': 'ICICI Bank Limited',
            'sector': 'Banking'
        },
        'SBIN': {
            'yahoo': 'SBIN.NS',
            'upstox': 'NSE_EQ|INE062A01020',
            'name': 'State Bank of India',
            'sector': 'Banking'
        },
        'AXISBANK': {
            'yahoo': 'AXISBANK.NS',
            'upstox': 'NSE_EQ|INE238A01034',
            'name': 'Axis Bank Limited',
            'sector': 'Banking'
        },
        'KOTAKBANK': {
            'yahoo': 'KOTAKBANK.NS',
            'upstox': 'NSE_EQ|INE237A01028',
            'name': 'Kotak Mahindra Bank Limited',
            'sector': 'Banking'
        },
        'INDUSINDBK': {
            'yahoo': 'INDUSINDBK.NS',
            'upstox': 'NSE_EQ|INE095A01012',
            'name': 'IndusInd Bank Limited',
            'sector': 'Banking'
        },
        'BANDHANBNK': {
            'yahoo': 'BANDHANBNK.NS',
            'upstox': 'NSE_EQ|INE545U01014',
            'name': 'Bandhan Bank Limited',
            'sector': 'Banking'
        },
        
        # IT Stocks
        'TCS': {
            'yahoo': 'TCS.NS',
            'upstox': 'NSE_EQ|INE467B01029',
            'name': 'Tata Consultancy Services Limited',
            'sector': 'Information Technology'
        },
        'INFY': {
            'yahoo': 'INFY.NS',
            'upstox': 'NSE_EQ|INE009A01021',
            'name': 'Infosys Limited',
            'sector': 'Information Technology'
        },
        'WIPRO': {
            'yahoo': 'WIPRO.NS',
            'upstox': 'NSE_EQ|INE075A01022',
            'name': 'Wipro Limited',
            'sector': 'Information Technology'
        },
        'HCLTECH': {
            'yahoo': 'HCLTECH.NS',
            'upstox': 'NSE_EQ|INE860A01027',
            'name': 'HCL Technologies Limited',
            'sector': 'Information Technology'
        },
        'TECHM': {
            'yahoo': 'TECHM.NS',
            'upstox': 'NSE_EQ|INE669C01036',
            'name': 'Tech Mahindra Limited',
            'sector': 'Information Technology'
        },
        
        # Oil & Gas
        'RELIANCE': {
            'yahoo': 'RELIANCE.NS',
            'upstox': 'NSE_EQ|INE002A01018',
            'name': 'Reliance Industries Limited',
            'sector': 'Oil Gas & Consumable Fuels'
        },
        'ONGC': {
            'yahoo': 'ONGC.NS',
            'upstox': 'NSE_EQ|INE213A01029',
            'name': 'Oil & Natural Gas Corporation Limited',
            'sector': 'Oil Gas & Consumable Fuels'
        },
        'BPCL': {
            'yahoo': 'BPCL.NS',
            'upstox': 'NSE_EQ|INE029A01011',
            'name': 'Bharat Petroleum Corporation Limited',
            'sector': 'Oil Gas & Consumable Fuels'
        },
        'IOC': {
            'yahoo': 'IOC.NS',
            'upstox': 'NSE_EQ|INE242A01010',
            'name': 'Indian Oil Corporation Limited',
            'sector': 'Oil Gas & Consumable Fuels'
        },
        
        # Automobiles
        'TATAMOTORS': {
            'yahoo': 'TATAMOTORS.NS',
            'upstox': 'NSE_EQ|INE155A01022',
            'name': 'Tata Motors Limited',
            'sector': 'Automobiles'
        },
        'M&M': {
            'yahoo': 'M&M.NS',
            'upstox': 'NSE_EQ|INE101A01026',
            'name': 'Mahindra & Mahindra Limited',
            'sector': 'Automobiles'
        },
        'MARUTI': {
            'yahoo': 'MARUTI.NS',
            'upstox': 'NSE_EQ|INE585B01010',
            'name': 'Maruti Suzuki India Limited',
            'sector': 'Automobiles'
        },
        'BAJAJ-AUTO': {
            'yahoo': 'BAJAJ-AUTO.NS',
            'upstox': 'NSE_EQ|INE917I01010',
            'name': 'Bajaj Auto Limited',
            'sector': 'Automobiles'
        },
        'EICHERMOT': {
            'yahoo': 'EICHERMOT.NS',
            'upstox': 'NSE_EQ|INE066A01021',
            'name': 'Eicher Motors Limited',
            'sector': 'Automobiles'
        },
        'HEROMOTOCO': {
            'yahoo': 'HEROMOTOCO.NS',
            'upstox': 'NSE_EQ|INE158A01026',
            'name': 'Hero MotoCorp Limited',
            'sector': 'Automobiles'
        },
        
        # FMCG
        'ITC': {
            'yahoo': 'ITC.NS',
            'upstox': 'NSE_EQ|INE154A01025',
            'name': 'ITC Limited',
            'sector': 'Fast Moving Consumer Goods'
        },
        'HINDUNILVR': {
            'yahoo': 'HINDUNILVR.NS',
            'upstox': 'NSE_EQ|INE030A01027',
            'name': 'Hindustan Unilever Limited',
            'sector': 'Fast Moving Consumer Goods'
        },
        'NESTLE': {
            'yahoo': 'NESTLE.NS',
            'upstox': 'NSE_EQ|INE239A01016',
            'name': 'Nestle India Limited',
            'sector': 'Fast Moving Consumer Goods'
        },
        'BRITANNIA': {
            'yahoo': 'BRITANNIA.NS',
            'upstox': 'NSE_EQ|INE216A01030',
            'name': 'Britannia Industries Limited',
            'sector': 'Fast Moving Consumer Goods'
        },
        
        # Pharma
        'SUNPHARMA': {
            'yahoo': 'SUNPHARMA.NS',
            'upstox': 'NSE_EQ|INE044A01036',
            'name': 'Sun Pharmaceutical Industries Limited',
            'sector': 'Pharmaceuticals'
        },
        'DRREDDY': {
            'yahoo': 'DRREDDY.NS',
            'upstox': 'NSE_EQ|INE089A01023',
            'name': 'Dr. Reddys Laboratories Limited',
            'sector': 'Pharmaceuticals'
        },
        'CIPLA': {
            'yahoo': 'CIPLA.NS',
            'upstox': 'NSE_EQ|INE059A01026',
            'name': 'Cipla Limited',
            'sector': 'Pharmaceuticals'
        },
        'DIVISLAB': {
            'yahoo': 'DIVISLAB.NS',
            'upstox': 'NSE_EQ|INE361B01024',
            'name': 'Divis Laboratories Limited',
            'sector': 'Pharmaceuticals'
        },
        
        # Metals & Mining
        'TATASTEEL': {
            'yahoo': 'TATASTEEL.NS',
            'upstox': 'NSE_EQ|INE081A01012',
            'name': 'Tata Steel Limited',
            'sector': 'Metals & Mining'
        },
        'JSWSTEEL': {
            'yahoo': 'JSWSTEEL.NS',
            'upstox': 'NSE_EQ|INE019A01038',
            'name': 'JSW Steel Limited',
            'sector': 'Metals & Mining'
        },
        'HINDALCO': {
            'yahoo': 'HINDALCO.NS',
            'upstox': 'NSE_EQ|INE038A01020',
            'name': 'Hindalco Industries Limited',
            'sector': 'Metals & Mining'
        },
        'NATIONALUM': {
            'yahoo': 'NATIONALUM.NS',
            'upstox': 'NSE_EQ|INE139A01034',
            'name': 'National Aluminium Company Limited',
            'sector': 'Metals & Mining'
        },
        
        # Telecom
        'BHARTIARTL': {
            'yahoo': 'BHARTIARTL.NS',
            'upstox': 'NSE_EQ|INE397D01024',
            'name': 'Bharti Airtel Limited',
            'sector': 'Diversified Telecommunication Services'
        },
        'IDEA': {
            'yahoo': 'IDEA.NS',
            'upstox': 'NSE_EQ|INE669E01016',
            'name': 'Vodafone Idea Limited',
            'sector': 'Diversified Telecommunication Services'
        },
        
        # Cement
        'ULTRACEMCO': {
            'yahoo': 'ULTRACEMCO.NS',
            'upstox': 'NSE_EQ|INE481G01011',
            'name': 'UltraTech Cement Limited',
            'sector': 'Construction Materials'
        },
        'SHREECEM': {
            'yahoo': 'SHREECEM.NS',
            'upstox': 'NSE_EQ|INE070A01015',
            'name': 'Shree Cement Limited',
            'sector': 'Construction Materials'
        },
        'GRASIM': {
            'yahoo': 'GRASIM.NS',
            'upstox': 'NSE_EQ|INE047A01021',
            'name': 'Grasim Industries Limited',
            'sector': 'Construction Materials'
        },
        
        # Power
        'NTPC': {
            'yahoo': 'NTPC.NS',
            'upstox': 'NSE_EQ|INE733E01010',
            'name': 'NTPC Limited',
            'sector': 'Independent Power and Renewable Electricity Producers'
        },
        'POWERGRID': {
            'yahoo': 'POWERGRID.NS',
            'upstox': 'NSE_EQ|INE752E01010',
            'name': 'Power Grid Corporation of India Limited',
            'sector': 'Electric Utilities'
        },
        
        # Real Estate
        'DLF': {
            'yahoo': 'DLF.NS',
            'upstox': 'NSE_EQ|INE271C01023',
            'name': 'DLF Limited',
            'sector': 'Real Estate Management & Development'
        },
        
        # Additional Popular Stocks
        'LT': {
            'yahoo': 'LT.NS',
            'upstox': 'NSE_EQ|INE018A01030',
            'name': 'Larsen & Toubro Limited',
            'sector': 'Capital Goods'
        },
        'ADANIPORTS': {
            'yahoo': 'ADANIPORTS.NS',
            'upstox': 'NSE_EQ|INE742F01042',
            'name': 'Adani Ports and Special Economic Zone Limited',
            'sector': 'Transportation Infrastructure'
        },
        'COALINDIA': {
            'yahoo': 'COALINDIA.NS',
            'upstox': 'NSE_EQ|INE522F01014',
            'name': 'Coal India Limited',
            'sector': 'Oil Gas & Consumable Fuels'
        },
        'BAJFINANCE': {
            'yahoo': 'BAJFINANCE.NS',
            'upstox': 'NSE_EQ|INE296A01024',
            'name': 'Bajaj Finance Limited',
            'sector': 'Consumer Finance'
        },
        'BAJAJFINSV': {
            'yahoo': 'BAJAJFINSV.NS',
            'upstox': 'NSE_EQ|INE918I01018',
            'name': 'Bajaj Finserv Limited',
            'sector': 'Diversified Financial Services'
        }
    }
    
    # Alternative names mapping
    ALIAS_MAP = {
        'TATA': 'TATAMOTORS',
        'HDFC': 'HDFCBANK',
        'ICICI': 'ICICIBANK',
        'AXIS': 'AXISBANK',
        'SBI': 'SBIN',
        'AIRTEL': 'BHARTIARTL',
        'RELIANCE': 'RELIANCE',
        'INFOSYS': 'INFY',
        'WIPRO': 'WIPRO'
    }
    
    @classmethod
    def get_symbol_info(cls, symbol: str) -> dict:
        """Get complete symbol information"""
        # Clean and normalize symbol
        symbol = symbol.upper().strip()
        
        # Check aliases first
        if symbol in cls.ALIAS_MAP:
            symbol = cls.ALIAS_MAP[symbol]
        
        # Return symbol info or create fallback
        if symbol in cls.SYMBOL_MAP:
            return cls.SYMBOL_MAP[symbol]
        else:
            # Create fallback mapping
            return {
                'yahoo': f'{symbol}.NS',
                'upstox': f'NSE_EQ|{symbol}',
                'name': f'{symbol} Limited',
                'sector': 'Unknown'
            }
    
    @classmethod
    def get_yahoo_symbol(cls, symbol: str) -> str:
        """Get Yahoo Finance symbol"""
        info = cls.get_symbol_info(symbol)
        return info['yahoo']
    
    @classmethod
    def get_upstox_symbol(cls, symbol: str) -> str:
        """Get Upstox symbol with ISIN"""
        info = cls.get_symbol_info(symbol)
        return info['upstox']
    
    @classmethod
    def is_valid_symbol(cls, symbol: str) -> bool:
        """Check if symbol is valid/supported"""
        symbol = symbol.upper().strip()
        return (symbol in cls.SYMBOL_MAP or 
                symbol in cls.ALIAS_MAP or
                len(symbol) >= 2)
    
    @classmethod
    def get_all_symbols(cls) -> list:
        """Get all supported symbols"""
        return list(cls.SYMBOL_MAP.keys())
    
    @classmethod
    def get_symbols_by_sector(cls, sector: str) -> list:
        """Get symbols by sector"""
        return [symbol for symbol, info in cls.SYMBOL_MAP.items() 
                if info['sector'].lower() == sector.lower()]
    
    @classmethod
    def search_symbols(cls, query: str) -> list:
        """Search symbols by name or symbol"""
        query = query.upper()
        results = []
        
        for symbol, info in cls.SYMBOL_MAP.items():
            if (query in symbol or 
                query in info['name'].upper() or
                query in info['sector'].upper()):
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'sector': info['sector']
                })
        
        return results
    @classmethod
    def test_symbol_mapping(cls):
        """Test the symbol mapping system"""
        print(f"\n🧪 SYMBOL MAPPING TEST")
        print(f"=" * 30)
        
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'ADANIPORTS', 'INFY']
        
        for symbol in test_symbols:
            info = cls.get_symbol_info(symbol)
            yahoo_sym = cls.get_yahoo_symbol(symbol)
            upstox_sym = cls.get_upstox_symbol(symbol)
            
            print(f"🏷️ {symbol}:")
            print(f"   Yahoo: {yahoo_sym}")
            print(f"   Upstox: {upstox_sym}")
            print(f"   Name: {info['name']}")
            print()
        
        print(f"✅ Total symbols supported: {len(cls.SYMBOL_MAP)}")
        return True

# Global instance
symbol_mapper = SymbolMapper()
