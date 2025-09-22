import streamlit as st
import os
import asyncio
import threading
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="DBR Trading Bot", 
    page_icon="🚀", 
    layout="centered"
)

# Main UI
st.title("🚀 DBR Trading Bot")
st.markdown("### Professional Stock Analysis Bot")

# Bot status display
col1, col2 = st.columns(2)
with col1:
    st.metric("Status", "🟢 Online")
with col2:
    st.metric("Platform", "Telegram")

st.markdown("""
## 📱 How to Use
1. **Open Telegram** and search for your bot
2. **Send** `/start` to begin  
3. **Enter** any stock name or symbol
4. **Receive** professional analysis instantly!

## 🎯 Features
- AI-Powered Symbol Resolution
- Multi-Timeframe Analysis
- Technical Indicators
- Price Targets & Risk Management
- Real-time NSE/BSE Data
""")

# Import and run the bot
@st.cache_resource
def start_telegram_bot():
    try:
        from bot import EnhancedTradingBot
        
        def run_bot():
            try:
                token = st.secrets["TELEGRAM_BOT_TOKEN"]
                bot = EnhancedTradingBot(token)
                asyncio.run(bot.run())
            except Exception as e:
                st.error(f"Bot Error: {e}")
        
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        return "Bot Started Successfully"
        
    except Exception as e:
        return f"Error: {e}"

# Auto-start bot
start_telegram_bot()

st.markdown("---")
st.markdown("**DBR Trading Bot** • Built with Streamlit & Python-Telegram-Bot")
