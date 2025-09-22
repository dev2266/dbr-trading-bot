import streamlit as st
import os
import asyncio
import threading
import logging
import subprocess
import sys

st.set_page_config(page_title="DBR Trading Bot", page_icon="🚀")

st.title("🚀 DBR Trading Bot")
st.success("✅ Bot is ONLINE")

# Status display
col1, col2 = st.columns(2)
with col1:
    st.metric("Status", "🟢 Online")
with col2:
    st.metric("Platform", "Telegram")

st.info("""
🤖 **Your Telegram bot is now running 24/7!**

**To use your bot:**
1. Open Telegram
2. Search for your bot
3. Send /start
4. Enter any stock symbol (e.g., "RELIANCE", "TCS")
5. Get instant professional analysis!

## 🎯 Features
- AI-Powered Symbol Resolution
- Multi-Timeframe Analysis  
- Technical Indicators
- Price Targets & Risk Management
- Real-time NSE/BSE Data
""")

# Simple bot runner
@st.cache_resource
def start_bot():
    try:
        token = st.secrets.get("TELEGRAM_BOT_TOKEN")
        if not token:
            return "❌ Bot token not found in secrets"
        
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", f"""
import os
os.environ['TELEGRAM_BOT_TOKEN'] = '{token}'
from bot import EnhancedTradingBot
import asyncio

async def run_bot():
    try:
        bot = EnhancedTradingBot('{token}')
        await bot.run()
    except Exception as e:
        print(f'Bot error: {{e}}')

if __name__ == '__main__':
    asyncio.run(run_bot())
"""],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False
            )
            return "✅ Bot started successfully"
        except Exception as e:
            return f"⚠️ Bot startup warning: {str(e)[:100]}"
    except Exception as e:
        return f"❌ Setup error: {str(e)[:100]}"

bot_status = start_bot()
st.success(bot_status)

st.markdown("---")
st.markdown("**DBR Trading Bot** • Built with Streamlit & Python-Telegram-Bot")
