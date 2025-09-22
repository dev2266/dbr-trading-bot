#!/usr/bin/env python3
"""
Azure App Service Integration for Enhanced Trading Bot
====================================================

Production-ready Flask app for Azure deployment with webhook support
"""

import os
import sys
import logging
import json
import asyncio
import threading
from datetime import datetime
from flask import Flask, request, jsonify

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import bot components
try:
    from bot import EnhancedTradingBot, config, db_manager, ptb_runner
    from telegram import Update
    BOT_IMPORTED = True
except ImportError as e:
    print(f"❌ Bot import failed: {e}")
    BOT_IMPORTED = False

# Azure App Service Configuration
AZURE_DEPLOYMENT = os.getenv('WEBSITE_SITE_NAME') is not None
IS_PRODUCTION = os.getenv('ENVIRONMENT') == 'production' or AZURE_DEPLOYMENT

# Configure logging for Azure
def setup_azure_logging():
    """Configure logging for Azure App Service"""
    log_level = logging.INFO if IS_PRODUCTION else logging.DEBUG
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler for Azure logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_azure_logging()

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'azure-trading-bot-secret')

# Global bot instance
trading_bot = None
bot_initialized = False

def initialize_bot():
    """Initialize the trading bot for Azure"""
    global trading_bot, bot_initialized
    
    try:
        if not BOT_IMPORTED:
            logger.error("❌ Bot components not imported properly")
            return False
            
        logger.info("🚀 Initializing Enhanced Trading Bot for Azure...")
        
        # Verify bot token
        if not config.BOT_TOKEN:
            logger.error("❌ TELEGRAM_BOT_TOKEN not configured")
            return False
            
        # Initialize bot with webhook mode
        trading_bot = EnhancedTradingBot(config.BOT_TOKEN)
        
        # Setup application but don't start polling
        trading_bot.setup_application()
        
        # Get or create persistent application
        application = ptb_runner.get_or_create_application(config.BOT_TOKEN)
        
        # Add all handlers to the application
        trading_bot._add_handlers_to_application(application)
        
        bot_initialized = True
        logger.info("✅ Enhanced Trading Bot initialized successfully for webhook mode")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bot initialization failed: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Azure"""
    try:
        # Check bot status
        bot_status = "initialized" if bot_initialized else "not_initialized"
        
        # Check database
        try:
            db_manager.execute_query("SELECT 1", fetch=True)
            db_status = "connected"
        except:
            db_status = "disconnected"
        
        # Check environment
        env_status = {
            'azure_deployment': AZURE_DEPLOYMENT,
            'production_mode': IS_PRODUCTION,
            'bot_token_configured': bool(config.BOT_TOKEN),
            'admin_users_configured': len(config.ADMIN_USER_IDS) > 0
        }
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'bot_status': bot_status,
            'database_status': db_status,
            'environment': env_status,
            'uptime': 'active'
        }
        
        logger.info("🏥 Health check passed")
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle Telegram webhook updates"""
    try:
        if not bot_initialized:
            logger.error("❌ Bot not initialized, cannot process webhook")
            return jsonify({'error': 'Bot not initialized'}), 500
            
        # Get update data
        update_data = request.get_json()
        
        if not update_data:
            logger.warning("⚠️ Empty webhook data received")
            return jsonify({'status': 'ignored'}), 200
            
        logger.info(f"📨 Webhook update received: {update_data.get('update_id', 'unknown')}")
        
        # Create Telegram Update object
        update = Update.de_json(update_data, ptb_runner._application.bot)
        
        if not update:
            logger.warning("⚠️ Invalid update data")
            return jsonify({'error': 'Invalid update'}), 400
            
        # Process update using persistent runner
        ptb_runner.process_update_sync(update)
        
        logger.info("✅ Webhook update processed successfully")
        return jsonify({'status': 'ok'}), 200
        
    except Exception as e:
        logger.error(f"❌ Webhook processing failed: {e}")
        return jsonify({'error': 'Webhook processing failed'}), 500

@app.route('/setwebhook', methods=['GET', 'POST'])
def set_webhook():
    """Set Telegram webhook URL"""
    try:
        if not config.BOT_TOKEN:
            return jsonify({'error': 'Bot token not configured'}), 500
            
        # Construct webhook URL
        app_name = os.getenv('WEBSITE_SITE_NAME')
        if app_name:
            webhook_url = f"https://{app_name}.azurewebsites.net/webhook"
        else:
            # Fallback for local testing
            webhook_url = f"{request.host_url}webhook"
            
        # Set webhook
        import requests
        response = requests.post(
            f"https://api.telegram.org/bot{config.BOT_TOKEN}/setWebhook",
            json={'url': webhook_url},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                logger.info(f"✅ Webhook set successfully: {webhook_url}")
                return jsonify({
                    'success': True,
                    'webhook_url': webhook_url,
                    'response': result
                }), 200
            else:
                logger.error(f"❌ Webhook setting failed: {result}")
                return jsonify({'error': result}), 400
        else:
            logger.error(f"❌ HTTP error setting webhook: {response.status_code}")
            return jsonify({'error': f'HTTP {response.status_code}'}), 500
            
    except Exception as e:
        logger.error(f"❌ Webhook setup failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhookinfo', methods=['GET'])
def webhook_info():
    """Get current webhook information"""
    try:
        if not config.BOT_TOKEN:
            return jsonify({'error': 'Bot token not configured'}), 500
            
        import requests
        response = requests.get(
            f"https://api.telegram.org/bot{config.BOT_TOKEN}/getWebhookInfo",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("📋 Webhook info retrieved")
            return jsonify(result), 200
        else:
            return jsonify({'error': f'HTTP {response.status_code}'}), 500
            
    except Exception as e:
        logger.error(f"❌ Webhook info failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deletewebhook', methods=['POST'])
def delete_webhook():
    """Delete current webhook (for testing)"""
    try:
        if not config.BOT_TOKEN:
            return jsonify({'error': 'Bot token not configured'}), 500
            
        import requests
        response = requests.post(
            f"https://api.telegram.org/bot{config.BOT_TOKEN}/deleteWebhook",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("🗑️ Webhook deleted")
            return jsonify(result), 200
        else:
            return jsonify({'error': f'HTTP {response.status_code}'}), 500
            
    except Exception as e:
        logger.error(f"❌ Webhook deletion failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get bot statistics"""
    try:
        if not bot_initialized:
            return jsonify({'error': 'Bot not initialized'}), 500
            
        # Get database stats
        try:
            total_users = db_manager.execute_query("SELECT COUNT(*) FROM users", fetch=True)
            total_analyses = db_manager.execute_query("SELECT COUNT(*) FROM analysis_history", fetch=True)
            active_users = db_manager.execute_query(
                "SELECT COUNT(*) FROM users WHERE last_activity > datetime('now', '-7 days')", 
                fetch=True
            )
            
            stats = {
                'total_users': total_users[0][0] if total_users else 0,
                'total_analyses': total_analyses[0][0] if total_analyses else 0,
                'active_users_7d': active_users[0][0] if active_users else 0,
                'bot_status': 'running',
                'deployment': 'azure',
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(stats), 200
            
        except Exception as db_error:
            logger.error(f"❌ Database stats failed: {db_error}")
            return jsonify({'error': 'Database unavailable'}), 500
            
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with bot information"""
    try:
        bot_status = "✅ Online" if bot_initialized else "❌ Offline"
        
        info = {
            'service': 'Enhanced Trading Bot API',
            'status': bot_status,
            'deployment': 'Azure App Service',
            'region': os.getenv('WEBSITE_SITE_NAME', 'local'),
            'environment': 'production' if IS_PRODUCTION else 'development',
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                'webhook': '/webhook (POST)',
                'health': '/health (GET)',
                'setwebhook': '/setwebhook (GET/POST)',
                'webhookinfo': '/webhookinfo (GET)',
                'stats': '/stats (GET)'
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"❌ Index endpoint failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize bot when module loads
if __name__ != '__main__':
    # This runs when imported by Gunicorn
    logger.info("🚀 Starting Enhanced Trading Bot on Azure App Service...")
    
    if initialize_bot():
        logger.info("✅ Bot initialized successfully - ready for webhooks")
    else:
        logger.error("❌ Bot initialization failed")

# For local testing
if __name__ == '__main__':
    logger.info("🧪 Running in local development mode...")
    
    if initialize_bot():
        logger.info("✅ Bot initialized - starting Flask development server")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
    else:
        logger.error("❌ Bot initialization failed - cannot start server")
        sys.exit(1)