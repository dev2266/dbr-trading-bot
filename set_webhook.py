#!/usr/bin/env python3
"""
Telegram Webhook Configuration Script
Run this after deploying your bot to Azure App Service
"""

import requests
import json
import os
from typing import Dict, Any

def set_telegram_webhook(bot_token: str, webhook_url: str) -> Dict[str, Any]:
    """Set Telegram webhook URL for your bot"""
    
    api_url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    
    payload = {
        "url": webhook_url,
        "drop_pending_updates": True,  # Clear any pending updates
        "allowed_updates": [
            "message", 
            "callback_query", 
            "inline_query"
        ]
    }
    
    try:
        print(f"🔧 Setting webhook URL: {webhook_url}")
        response = requests.post(api_url, json=payload, timeout=30)
        result = response.json()
        
        if result.get("ok"):
            print("✅ Webhook set successfully!")
            print(f"📋 Response: {result}")
            return result
        else:
            print(f"❌ Failed to set webhook: {result}")
            return result
            
    except Exception as e:
        print(f"❌ Error setting webhook: {e}")
        return {"ok": False, "error": str(e)}

def get_webhook_info(bot_token: str) -> Dict[str, Any]:
    """Get current webhook information"""
    
    api_url = f"https://api.telegram.org/bot{bot_token}/getWebhookInfo"
    
    try:
        response = requests.get(api_url, timeout=30)
        result = response.json()
        
        if result.get("ok"):
            webhook_info = result.get("result", {})
            print("📊 Current Webhook Information:")
            print(f"   URL: {webhook_info.get('url', 'Not set')}")
            print(f"   Pending Updates: {webhook_info.get('pending_update_count', 0)}")
            print(f"   Last Error: {webhook_info.get('last_error_message', 'None')}")
            print(f"   Max Connections: {webhook_info.get('max_connections', 0)}")
            return result
        else:
            print(f"❌ Failed to get webhook info: {result}")
            return result
            
    except Exception as e:
        print(f"❌ Error getting webhook info: {e}")
        return {"ok": False, "error": str(e)}

def test_webhook_endpoint(webhook_url: str) -> bool:
    """Test if webhook endpoint is accessible"""
    
    try:
        print(f"🧪 Testing webhook endpoint: {webhook_url}")
        response = requests.post(webhook_url, json={}, timeout=10)
        
        if response.status_code == 200:
            print("✅ Webhook endpoint is accessible")
            return True
        else:
            print(f"⚠️ Webhook responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to webhook endpoint - check if your Azure app is running")
        return False
    except requests.exceptions.Timeout:
        print("❌ Webhook endpoint timeout - check Azure app performance")
        return False
    except Exception as e:
        print(f"❌ Error testing webhook: {e}")
        return False

def main():
    """Main function to configure webhook"""
    
    print("🤖 TELEGRAM WEBHOOK CONFIGURATION")
    print("=" * 50)
    
    # Configuration
    BOT_TOKEN = input("Enter your Telegram Bot Token: ").strip()
    
    if not BOT_TOKEN:
        print("❌ Bot token is required!")
        return
    
    # Your Azure App Service URL
    AZURE_APP_NAME = input("Enter your Azure App Service name (e.g., 'dbrx'): ").strip()
    
    if not AZURE_APP_NAME:
        print("❌ Azure app name is required!")
        return
    
    WEBHOOK_URL = f"https://{AZURE_APP_NAME}.azurewebsites.net/webhook"
    
    print(f"\n🔧 Configuration:")
    print(f"   Bot Token: {BOT_TOKEN[:10]}...")
    print(f"   Azure App: {AZURE_APP_NAME}")
    print(f"   Webhook URL: {WEBHOOK_URL}")
    
    # Step 1: Test webhook endpoint
    print(f"\n📋 Step 1: Testing webhook endpoint...")
    if not test_webhook_endpoint(WEBHOOK_URL):
        print("❌ Webhook endpoint test failed!")
        print("💡 Make sure your Azure App Service is running and deployed correctly")
        return
    
    # Step 2: Get current webhook info
    print(f"\n📋 Step 2: Checking current webhook configuration...")
    get_webhook_info(BOT_TOKEN)
    
    # Step 3: Set new webhook
    print(f"\n📋 Step 3: Setting new webhook...")
    result = set_telegram_webhook(BOT_TOKEN, WEBHOOK_URL)
    
    if result.get("ok"):
        print(f"\n🎉 SUCCESS! Your bot is now configured for production!")
        print(f"🌐 Webhook URL: {WEBHOOK_URL}")
        print(f"✅ Your bot should now respond to messages")
        
        # Final verification
        print(f"\n📋 Final verification...")
        get_webhook_info(BOT_TOKEN)
        
        print(f"\n💡 Test your bot by sending: /start")
        
    else:
        print(f"\n❌ Failed to set webhook. Check the error above.")

if __name__ == "__main__":
    main()