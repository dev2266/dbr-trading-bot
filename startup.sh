#!/bin/bash

# Azure App Service startup script for Enhanced Trading Bot
echo "Starting Enhanced Trading Bot on Azure..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot"
export ENVIRONMENT="production"

# Create necessary directories
mkdir -p /home/site/wwwroot/logs
mkdir -p /home/site/wwwroot/data

# Set permissions
chmod -R 755 /home/site/wwwroot

# Initialize database and start Flask app
cd /home/site/wwwroot

# Check if bot token exists
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "ERROR: TELEGRAM_BOT_TOKEN not set!"
    exit 1
fi

echo "✅ Environment configured for Azure deployment"
echo "✅ Bot token configured"
echo "✅ Starting Flask webhook server..."

# Start the Flask application with Gunicorn
exec gunicorn --bind 0.0.0.0:8000 --workers 1 --worker-class sync --timeout 120 --preload --access-logfile - --error-logfile - azure_app:app