#!/usr/bin/env python3
"""
Enhanced Azure startup script with all features
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Azure environment setup
        os.environ['PYTHONPATH'] = '/home/site/wwwroot'
        os.environ['ENVIRONMENT'] = 'production'
        
        # Create required directories
        dirs_to_create = [
            '/home/site/wwwroot/logs',
            '/home/site/wwwroot/data',
            '/home/site/wwwroot/temp',
            '/home/site/wwwroot/cache',
            '/home/site/wwwroot/models'  # For ML models
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")
        
        # Verify essential modules
        try:
            import your_analysis_module
            logger.info("✅ Professional analysis module loaded")
        except ImportError:
            logger.warning("⚠️ Professional analysis module not found")
        
        try:
            import symbol_mapper
            logger.info("✅ Symbol mapper loaded")
        except ImportError:
            logger.warning("⚠️ Symbol mapper not found")
            
        try:
            import upstox_fetcher
            logger.info("✅ Upstox fetcher loaded")
        except ImportError:
            logger.warning("⚠️ Upstox fetcher not found")
            
        try:
            import perplexity_symbol_resolver
            logger.info("✅ Perplexity resolver loaded")
        except ImportError:
            logger.warning("⚠️ Perplexity resolver not found")
        
        logger.info("🚀 Starting Enhanced Trading Bot with all features...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Import and run bot
        from bot import main as bot_main
        bot_main()
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
