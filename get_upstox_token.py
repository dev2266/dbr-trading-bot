#!/usr/bin/env python3
"""
Complete Upstox Token Generator with Auto .env Save
==================================================
- Manual and Semi-Automated Token Generation  
- Automatic .env File Update
- Full Error Handling
- Regulatory Compliant Approach

IMPORTANT: Upstox doesn't recommend full automation due to SEBI guidelines.
"""

import os
import sys
import requests
import webbrowser
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import json
import time
from dotenv import load_dotenv, set_key, find_dotenv
from pathlib import Path

# Optional: Selenium for automation (use with caution)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available. Manual mode only.")

class UpstoxTokenGenerator:
    """Complete Upstox Token Generator with Auto .env Save"""
    
    def __init__(self, api_key=None, api_secret=None, redirect_uri=None):
        # Load existing .env if available
        load_dotenv()
        
        # Get credentials from .env or parameters
        self.api_key = api_key or os.getenv('UPSTOX_API_KEY')
        self.api_secret = api_secret or os.getenv('UPSTOX_API_SECRET') 
        self.redirect_uri = redirect_uri or os.getenv('UPSTOX_REDIRECT_URI', 'https://127.0.0.1:8000/')
        
        # API endpoints
        self.auth_url_base = "https://api.upstox.com/v2/login/authorization/dialog"
        self.token_url = "https://api.upstox.com/v2/login/authorization/token"
        
        # Validate credentials
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate required credentials"""
        missing = []
        if not self.api_key:
            missing.append("UPSTOX_API_KEY")
        if not self.api_secret:
            missing.append("UPSTOX_API_SECRET")
        
        if missing:
            print(f"\n❌ Missing required credentials: {', '.join(missing)}")
            print("\n📝 Please provide them via:")
            print("1. .env file")
            print("2. Environment variables")
            print("3. Constructor parameters")
            sys.exit(1)
    
    def generate_auth_url(self):
        """Generate authorization URL"""
        auth_url = (
            f"{self.auth_url_base}?"
            f"response_type=code&"
            f"client_id={self.api_key}&"
            f"redirect_uri={self.redirect_uri}"
        )
        return auth_url
    
    def manual_token_generation(self):
        """Manual token generation - RECOMMENDED APPROACH"""
        print("\n🔐 MANUAL TOKEN GENERATION (RECOMMENDED)")
        print("=" * 50)
        
        # Generate and display auth URL
        auth_url = self.generate_auth_url()
        print(f"\n1. 🌐 Opening authorization URL in browser...")
        print(f"   URL: {auth_url}")
        
        # Open browser automatically
        try:
            webbrowser.open(auth_url)
            print("   ✅ Browser opened automatically")
        except Exception as e:
            print(f"   ⚠️ Could not open browser automatically: {e}")
            print(f"   📋 Please copy and paste the URL manually")
        
        print(f"\n2. 🔑 Complete authentication in browser:")
        print(f"   - Enter mobile number")
        print(f"   - Enter OTP") 
        print(f"   - Enter PIN")
        print(f"   - Grant permissions")
        
        print(f"\n3. 📋 After authentication, copy the authorization code:")
        print(f"   - You'll be redirected to: {self.redirect_uri}?code=XXXXXXX")
        print(f"   - Copy the 'code' parameter value")
        
        # Get auth code from user
        print(f"\n" + "="*50)
        auth_code = input("4. 🔑 Paste authorization code here: ").strip()
        
        if not auth_code:
            print("❌ No authorization code provided!")
            return None
        
        return self._exchange_code_for_token(auth_code)
    
    def _exchange_code_for_token(self, auth_code):
        """Exchange authorization code for access token"""
        print(f"\n🔄 Exchanging authorization code for access token...")
        
        try:
            headers = {
                'accept': 'application/json',
                'Api-Version': '2.0',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'code': auth_code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            print(f"📡 Making API request to Upstox...")
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            if response.status_code == 200:
                token_data = response.json()
                
                if 'access_token' in token_data:
                    access_token = token_data['access_token']
                    print(f"✅ Access token generated successfully!")
                    
                    # Save to .env file
                    self._save_token_to_env(access_token, token_data)
                    
                    return {
                        'access_token': access_token,
                        'token_data': token_data,
                        'generated_at': datetime.now().isoformat()
                    }
                else:
                    print(f"❌ No access token in response: {token_data}")
                    return None
                    
            else:
                print(f"❌ Token exchange failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Token exchange error: {e}")
            return None
    
    def _save_token_to_env(self, access_token, token_data):
        """Automatically save token to .env file"""
        print(f"\n💾 Saving token to .env file...")
        
        try:
            # Find or create .env file
            env_path = find_dotenv()
            if not env_path:
                env_path = '.env'
                Path(env_path).touch()
            
            # Save all credentials
            set_key(env_path, 'UPSTOX_API_KEY', self.api_key)
            set_key(env_path, 'UPSTOX_API_SECRET', self.api_secret)
            set_key(env_path, 'UPSTOX_REDIRECT_URI', self.redirect_uri)
            set_key(env_path, 'UPSTOX_ACCESS_TOKEN', access_token)
            
            # Add timestamp and additional info
            set_key(env_path, 'UPSTOX_TOKEN_GENERATED_AT', datetime.now().isoformat())
            
            if 'token_type' in token_data:
                set_key(env_path, 'UPSTOX_TOKEN_TYPE', token_data['token_type'])
            
            if 'expires_in' in token_data:
                set_key(env_path, 'UPSTOX_TOKEN_EXPIRES_IN', str(token_data['expires_in']))
            
            print(f"✅ Token saved to: {os.path.abspath(env_path)}")
            print(f"🔑 Access token: {access_token[:20]}...{access_token[-10:]}")
            
            # Display .env file contents
            self._display_env_contents(env_path)
            
        except Exception as e:
            print(f"❌ Failed to save token to .env: {e}")
    
    def _display_env_contents(self, env_path):
        """Display current .env file contents"""
        print(f"\n📄 Current .env file contents:")
        print(f"=" * 40)
        
        try:
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Mask sensitive values
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if 'TOKEN' in key and len(value) > 20:
                                masked_value = f"{value[:10]}...{value[-5:]}"
                                print(f"{line_num:2d}. {key}={masked_value}")
                            else:
                                print(f"{line_num:2d}. {line}")
                        else:
                            print(f"{line_num:2d}. {line}")
        except Exception as e:
            print(f"❌ Could not read .env file: {e}")
    
    def validate_token(self, access_token):
        """Validate generated token"""
        print(f"\n🔍 Validating generated token...")
        
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            # Test with user profile endpoint
            response = requests.get(
                'https://api.upstox.com/v2/user/profile',
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    user_name = data.get('data', {}).get('user_name', 'Unknown')
                    print(f"✅ Token validation successful!")
                    print(f"👤 User: {user_name}")
                    return True
                else:
                    print(f"❌ Token validation failed: {data}")
                    return False
            else:
                print(f"❌ Token validation failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Token validation error: {e}")
            return False
    
    def semi_automated_generation(self):
        """Semi-automated generation using selenium (use with caution)"""
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available. Install: pip install selenium")
            return None
        
        print("\n🤖 SEMI-AUTOMATED TOKEN GENERATION")
        print("⚠️ WARNING: Use with caution - regulatory concerns")
        print("=" * 50)
        
        # Get user credentials
        mobile_no = input("📱 Enter mobile number: ").strip()
        pin = input("🔐 Enter PIN: ").strip()
        
        if not mobile_no or not pin:
            print("❌ Mobile number and PIN required!")
            return None
        
        try:
            # Setup Chrome driver
            options = Options()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Note: Headless mode often fails due to CORS issues
            # options.add_argument('--headless')
            
            print("🌐 Opening browser for authentication...")
            driver = webdriver.Chrome(options=options)
            
            # Navigate to auth URL
            auth_url = self.generate_auth_url()
            driver.get(auth_url)
            
            wait = WebDriverWait(driver, 30)
            
            # Enter mobile number
            print("📱 Entering mobile number...")
            mobile_input = wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="text"]')))
            mobile_input.send_keys(mobile_no)
            
            # Click get OTP
            get_otp_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="getOtp"]')))
            get_otp_btn.click()
            
            # Wait for OTP input and let user enter manually
            print("📨 OTP sent! Please enter OTP in the browser window...")
            print("⏳ Waiting for OTP entry and PIN...")
            
            # Wait for redirect (indicates successful auth)
            print("⏳ Waiting for authentication completion...")
            wait.until(lambda driver: self.redirect_uri in driver.current_url)
            
            # Extract authorization code
            current_url = driver.current_url
            parsed_url = urlparse(current_url)
            code = parse_qs(parsed_url.query).get('code', [None])[0]
            
            driver.quit()
            
            if code:
                print(f"✅ Authorization code extracted: {code[:10]}...{code[-5:]}")
                return self._exchange_code_for_token(code)
            else:
                print("❌ No authorization code found in URL")
                return None
                
        except Exception as e:
            print(f"❌ Semi-automated generation failed: {e}")
            try:
                driver.quit()
            except:
                pass
            return None

def main():
    """Main function"""
    print("🚀 UPSTOX TOKEN GENERATOR WITH AUTO .env SAVE")
    print("=" * 55)
    
    # Initialize generator
    try:
        generator = UpstoxTokenGenerator()
    except SystemExit:
        return
    
    print(f"\n📋 Available Methods:")
    print(f"1. 🔑 Manual Generation (RECOMMENDED)")
    print(f"2. 🤖 Semi-Automated Generation (Selenium)")
    print(f"3. ❌ Exit")
    
    while True:
        try:
            choice = input(f"\n👉 Select method (1-3): ").strip()
            
            if choice == '1':
                result = generator.manual_token_generation()
                break
            elif choice == '2':
                result = generator.semi_automated_generation()
                break
            elif choice == '3':
                print("👋 Goodbye!")
                return
            else:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user.")
            return
    
    # Validate generated token
    if result and 'access_token' in result:
        print(f"\n" + "="*50)
        print(f"🎉 TOKEN GENERATION SUCCESSFUL!")
        print(f"="*50)
        
        # Validate token
        if generator.validate_token(result['access_token']):
            print(f"\n✅ Your Upstox API is now ready to use!")
            print(f"💡 Token expires at 3:30 AM daily - regenerate as needed")
            print(f"🔄 Run this script again tomorrow to get a fresh token")
            
            # Show next steps
            print(f"\n🚀 NEXT STEPS:")
            print(f"1. Run: python get_upstox_token.py (to test connection)")
            print(f"2. Run: python bot.py (to start your trading bot)")
        else:
            print(f"\n⚠️ Token generated but validation failed.")
            print(f"💡 Try regenerating the token.")
    else:
        print(f"\n❌ TOKEN GENERATION FAILED!")
        print(f"💡 Please check your credentials and try again.")

if __name__ == "__main__":
    main()
