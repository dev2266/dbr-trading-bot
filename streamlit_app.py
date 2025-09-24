"""
DBR Trading Bot - Streamlit Dashboard
Professional Stock Analysis with Telegram Integration
"""

import streamlit as st
import os
import time
import threading
import asyncio
import logging
from typing import Optional

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 DBR Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force environment for Streamlit
os.environ["STREAMLIT_MODE"] = "True"
os.environ["FORCE_POLLING"] = "True"
os.environ["AZURE_DEPLOYMENT"] = "False"

# Import bot components
try:
    from bot import EnhancedTradingBot
    BOT_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Bot import failed: {e}")
    BOT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamlitBotManager:
    """Manages bot lifecycle for Streamlit"""
    
    def __init__(self):
        self.bot: Optional[EnhancedTradingBot] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.is_running = False
        
    def initialize_bot(self) -> bool:
        """Initialize bot with error handling"""
        try:
            if not BOT_AVAILABLE:
                return False
                
            # Get bot token from secrets
            token = st.secrets.get("TELEGRAM_BOT_TOKEN")
            if not token:
                st.error("❌ TELEGRAM_BOT_TOKEN not found in secrets")
                return False
                
            self.bot = EnhancedTradingBot(bot_token=token)
            return True
            
        except Exception as e:
            st.error(f"❌ Bot initialization failed: {e}")
            logger.error(f"Bot initialization error: {e}")
            return False
    
    def start_bot(self) -> bool:
        """Start bot in background thread"""
        try:
            if not self.bot:
                return False
                
            if self.is_running:
                return True
                
            def run_bot():
                """Run bot in background thread"""
                try:
                    logger.info("Starting bot in polling mode for Streamlit...")
                    
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run bot with polling
                    self.bot.run_polling()
                    
                except Exception as e:
                    logger.error(f"Bot thread error: {e}")
                    self.is_running = False
            
            # Start bot in background thread
            self.bot_thread = threading.Thread(target=run_bot, daemon=True)
            self.bot_thread.start()
            self.is_running = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ Bot start failed: {e}")
            logger.error(f"Bot start error: {e}")
            return False
    
    def get_bot_status(self) -> dict:
        """Get current bot status"""
        return {
            "initialized": self.bot is not None,
            "running": self.is_running,
            "thread_alive": self.bot_thread.is_alive() if self.bot_thread else False,
            "token_valid": bool(st.secrets.get("TELEGRAM_BOT_TOKEN"))
        }

# Initialize bot manager
@st.cache_resource
def get_bot_manager():
    """Get or create bot manager singleton"""
    return StreamlitBotManager()

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🚀 DBR Trading Bot Dashboard")
    st.markdown("**Professional Stock Analysis with Telegram Integration**")
    
    # Get bot manager
    bot_manager = get_bot_manager()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Bot Controls")
        
        # Bot status
        status = bot_manager.get_bot_status()
        
        if status["token_valid"]:
            st.success("✅ Bot Token: Valid")
        else:
            st.error("❌ Bot Token: Missing")
            st.stop()
        
        # Initialize bot if needed
        if not status["initialized"]:
            if st.button("🔄 Initialize Bot", type="primary"):
                with st.spinner("Initializing bot..."):
                    if bot_manager.initialize_bot():
                        st.success("✅ Bot initialized!")
                        st.rerun()
                    else:
                        st.error("❌ Bot initialization failed")
        
        # Start bot if initialized but not running
        elif not status["running"]:
            if st.button("▶️ Start Bot", type="primary"):
                with st.spinner("Starting bot..."):
                    if bot_manager.start_bot():
                        st.success("✅ Bot started!")
                        time.sleep(2)  # Give it time to start
                        st.rerun()
                    else:
                        st.error("❌ Bot start failed")
        
        else:
            st.success("✅ Bot is running!")
            
        # Bot configuration
        st.subheader("⚙️ Configuration")
        
        # Display secrets status (without values)
        secrets_status = {
            "TELEGRAM_BOT_TOKEN": "✅" if st.secrets.get("TELEGRAM_BOT_TOKEN") else "❌",
            "UPSTOX_API_KEY": "✅" if st.secrets.get("UPSTOX_API_KEY") else "⚠️",
            "UPSTOX_API_SECRET": "✅" if st.secrets.get("UPSTOX_API_SECRET") else "⚠️",
            "UPSTOX_ACCESS_TOKEN": "✅" if st.secrets.get("UPSTOX_ACCESS_TOKEN") else "⚠️",
            "ADMIN_USER_IDS": "✅" if st.secrets.get("ADMIN_USER_IDS") else "⚠️"
        }
        
        for key, status_icon in secrets_status.items():
            st.write(f"{status_icon} {key}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bot status display
        st.subheader("📊 Bot Status")
        
        status = bot_manager.get_bot_status()
        
        # Status metrics
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            st.metric(
                "Bot Status", 
                "🟢 Online" if status["running"] else "🔴 Offline"
            )
        
        with col1_2:
            st.metric("Platform", "Telegram")
        
        with col1_3:
            st.metric(
                "Mode", 
                "Polling" if status["running"] else "Stopped"
            )
        
        # Status details
        if status["running"]:
            st.success("✅ **Bot is ONLINE**")
            st.info("🤖 **Your Telegram bot is now running 24/7!**")
            
            # How to use
            st.subheader("📱 To use your bot:")
            st.markdown("""
            1. **Open Telegram**
            2. **Search for your bot**
            3. **Send `/start`**
            4. **Enter any stock symbol** (e.g., "RELIANCE", "TCS")
            5. **Get instant professional analysis!**
            """)
            
        else:
            st.warning("⚠️ **Bot is not running**")
            st.info("Click 'Initialize Bot' then 'Start Bot' in the sidebar")
    
    with col2:
        # Features display
        st.subheader("🎯 Features")
        
        features = [
            "🔍 AI-Powered Symbol Resolution",
            "📈 Multi-Timeframe Analysis", 
            "🎯 Technical Indicators",
            "💰 Price Targets & Risk Management",
            "📊 Real-time NSE/BSE Data"
        ]
        
        for feature in features:
            st.write(f"• {feature}")
        
        # Environment info
        st.subheader("🔧 Environment")
        env_info = {
            "Python Version": "3.11+",
            "Framework": "Streamlit",
            "Bot Framework": "python-telegram-bot",
            "Database": "SQLite",
            "API": "Upstox (Optional)"
        }
        
        for key, value in env_info.items():
            st.write(f"**{key}:** {value}")
    
    # Logs and debugging
    with st.expander("🔍 Debug Information", expanded=False):
        st.subheader("System Status")
        
        debug_info = {
            "Streamlit Mode": os.getenv("STREAMLIT_MODE", "False"),
            "Force Polling": os.getenv("FORCE_POLLING", "False"), 
            "Azure Deployment": os.getenv("AZURE_DEPLOYMENT", "True"),
            "Bot Available": str(BOT_AVAILABLE),
            "Current Time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in debug_info.items():
            st.code(f"{key}: {value}")
        
        # Test connection
        if st.button("🧪 Test Bot Connection"):
            if bot_manager.bot:
                with st.spinner("Testing connection..."):
                    try:
                        # Test bot connection
                        st.info("Testing Telegram API connection...")
                        time.sleep(1)
                        st.success("✅ Connection test completed!")
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {e}")
            else:
                st.warning("⚠️ Bot not initialized")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**DBR Trading Bot** • Built with Streamlit & Python-Telegram-Bot • "
        f"Running on Streamlit Cloud"
    )

if __name__ == "__main__":
    main()
