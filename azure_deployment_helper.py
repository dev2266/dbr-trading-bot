#!/usr/bin/env python3
"""
Azure Deployment Helper Script
=============================

Helps deploy and manage the Enhanced Trading Bot on Azure
"""

import os
import sys
import json
import requests
import subprocess
from datetime import datetime

class AzureDeploymentHelper:
    """Helper class for Azure deployment tasks"""
    
    def __init__(self, app_name=None, bot_token=None):
        self.app_name = app_name or os.getenv('AZURE_APP_NAME')
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.base_url = f"https://{self.app_name}.azurewebsites.net" if self.app_name else None
        
    def check_environment(self):
        """Check if environment is properly configured"""
        print("🔍 Checking Azure deployment environment...")
        
        issues = []
        
        if not self.app_name:
            issues.append("❌ Azure App Name not configured (set AZURE_APP_NAME)")
        else:
            print(f"✅ Azure App Name: {self.app_name}")
            
        if not self.bot_token:
            issues.append("❌ Bot Token not configured (set TELEGRAM_BOT_TOKEN)")
        else:
            print(f"✅ Bot Token: {self.bot_token[:10]}...")
            
        if not self.base_url:
            issues.append("❌ Base URL cannot be constructed")
        else:
            print(f"✅ Base URL: {self.base_url}")
            
        if issues:
            print("\n⚠️ Issues found:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        print("✅ Environment check passed!")
        return True
    
    def test_health(self):
        """Test the health endpoint"""
        if not self.base_url:
            print("❌ Cannot test health - no base URL")
            return False
            
        try:
            print(f"🏥 Testing health endpoint: {self.base_url}/health")
            response = requests.get(f"{self.base_url}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed!")
                print(f"   Status: {data.get('status')}")
                print(f"   Bot Status: {data.get('bot_status')}")
                print(f"   Database: {data.get('database_status')}")
                return True
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def set_webhook(self):
        """Set the Telegram webhook"""
        if not self.bot_token or not self.base_url:
            print("❌ Cannot set webhook - missing token or URL")
            return False
            
        try:
            webhook_url = f"{self.base_url}/webhook"
            print(f"🔗 Setting webhook: {webhook_url}")
            
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/setWebhook",
                json={'url': webhook_url},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    print("✅ Webhook set successfully!")
                    print(f"   URL: {webhook_url}")
                    return True
                else:
                    print(f"❌ Webhook error: {data}")
                    return False
            else:
                print(f"❌ HTTP error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Webhook setup error: {e}")
            return False
    
    def get_webhook_info(self):
        """Get current webhook information"""
        if not self.bot_token:
            print("❌ Cannot get webhook info - missing token")
            return False
            
        try:
            print("📋 Getting webhook information...")
            response = requests.get(
                f"https://api.telegram.org/bot{self.bot_token}/getWebhookInfo",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    result = data.get('result', {})
                    print("✅ Webhook info:")
                    print(f"   URL: {result.get('url', 'Not set')}")
                    print(f"   Has Custom Certificate: {result.get('has_custom_certificate', False)}")
                    print(f"   Pending Updates: {result.get('pending_update_count', 0)}")
                    if result.get('last_error_date'):
                        error_date = datetime.fromtimestamp(result['last_error_date'])
                        print(f"   Last Error: {result.get('last_error_message')} ({error_date})")
                    return True
                else:
                    print(f"❌ Error: {data}")
                    return False
            else:
                print(f"❌ HTTP error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Webhook info error: {e}")
            return False
    
    def test_webhook(self):
        """Test webhook endpoint directly"""
        if not self.base_url:
            print("❌ Cannot test webhook - no base URL")
            return False
            
        try:
            print(f"📨 Testing webhook endpoint: {self.base_url}/webhook")
            # Send a test POST request (empty payload)
            response = requests.post(f"{self.base_url}/webhook", json={}, timeout=10)
            
            print(f"   Response: HTTP {response.status_code}")
            if response.text:
                print(f"   Body: {response.text}")
                
            return response.status_code in [200, 400] # 400 is OK for empty payload
            
        except Exception as e:
            print(f"❌ Webhook test error: {e}")
            return False
    
    def get_app_stats(self):
        """Get application statistics"""
        if not self.base_url:
            print("❌ Cannot get stats - no base URL")
            return False
            
        try:
            print(f"📊 Getting app statistics: {self.base_url}/stats")
            response = requests.get(f"{self.base_url}/stats", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ App Statistics:")
                print(f"   Total Users: {data.get('total_users', 0)}")
                print(f"   Total Analyses: {data.get('total_analyses', 0)}")
                print(f"   Active Users (7d): {data.get('active_users_7d', 0)}")
                print(f"   Bot Status: {data.get('bot_status', 'unknown')}")
                return True
            else:
                print(f"❌ Stats error: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return False
    
    def run_full_test(self):
        """Run complete deployment test"""
        print("🚀 Running full Azure deployment test...")
        print("=" * 50)
        
        tests = [
            ("Environment Check", self.check_environment),
            ("Health Check", self.test_health),
            ("Webhook Test", self.test_webhook),
            ("Get Webhook Info", self.get_webhook_info),
            ("Set Webhook", self.set_webhook),
            ("App Statistics", self.get_app_stats)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}:")
            print("-" * 30)
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your bot is ready for production.")
        else:
            print("⚠️  Some tests failed. Check the issues above.")
        
        return passed == total

def main():
    """Main function"""
    print("🔧 Azure Deployment Helper for Enhanced Trading Bot")
    print("=" * 55)
    
    # Get configuration
    app_name = input("Enter your Azure App Name (e.g., 'my-trading-bot'): ").strip()
    if not app_name:
        print("❌ App name is required")
        return
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        bot_token = input("Enter your Telegram Bot Token: ").strip()
        if not bot_token:
            print("❌ Bot token is required")
            return
    
    # Create helper instance
    helper = AzureDeploymentHelper(app_name, bot_token)
    
    # Show menu
    while True:
        print("\n🔧 Azure Deployment Helper")
        print("1. Run full test")
        print("2. Check health")
        print("3. Set webhook")
        print("4. Get webhook info")
        print("5. Test webhook")
        print("6. Get app stats")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            helper.run_full_test()
        elif choice == '2':
            helper.test_health()
        elif choice == '3':
            helper.set_webhook()
        elif choice == '4':
            helper.get_webhook_info()
        elif choice == '5':
            helper.test_webhook()
        elif choice == '6':
            helper.get_app_stats()
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

if __name__ == '__main__':
    main()