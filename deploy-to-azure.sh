#!/bin/bash

# Azure Deployment Script for Enhanced Trading Bot
# Bash script for Linux/Mac users

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Azure Deployment Script for Enhanced Trading Bot${NC}"
echo -e "${GREEN}======================================================${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found. Please install Azure CLI first.${NC}"
    echo -e "${YELLOW}Install from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Azure CLI found${NC}"

# Check if logged in
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}🔑 Please login to Azure...${NC}"
    az login
fi

ACCOUNT_NAME=$(az account show --query user.name -o tsv)
echo -e "${GREEN}✅ Logged in as: $ACCOUNT_NAME${NC}"

# Get configuration
if [ -z "$1" ]; then
    echo -n "Enter your Azure App Service name (globally unique): "
    read APP_NAME
else
    APP_NAME="$1"
fi

if [ -z "$APP_NAME" ]; then
    echo -e "${RED}❌ App name is required${NC}"
    exit 1
fi

RESOURCE_GROUP="${2:-trading-bot-rg}"
LOCATION="${3:-Southeast Asia}"

echo -e "${CYAN}📝 Configuration:${NC}"
echo -e "   App Name: ${APP_NAME}"
echo -e "   Resource Group: ${RESOURCE_GROUP}"
echo -e "   Location: ${LOCATION}"

# Create resource group
echo -e "${YELLOW}📦 Creating resource group...${NC}"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output table || true

# Create App Service Plan
echo -e "${YELLOW}📋 Creating App Service Plan...${NC}"
az appservice plan create --name "${APP_NAME}-plan" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --sku B1 --is-linux --output table || true

# Create Web App
echo -e "${YELLOW}🌐 Creating Web App...${NC}"
if ! az webapp create --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --plan "${APP_NAME}-plan" --runtime "PYTHON|3.11" --output table; then
    echo -e "${RED}❌ Web App creation failed${NC}"
    exit 1
fi

# Get Telegram Bot Token
BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
if [ -z "$BOT_TOKEN" ]; then
    echo -n "Enter your Telegram Bot Token: "
    read -s BOT_TOKEN
    echo
fi

if [ -z "$BOT_TOKEN" ]; then
    echo -e "${RED}❌ Bot token is required${NC}"
    exit 1
fi

# Get Admin User ID
echo -n "Enter your Telegram User ID (admin): "
read ADMIN_USER_ID
if [ -z "$ADMIN_USER_ID" ]; then
    ADMIN_USER_ID="123456789"
fi

# Configure app settings
echo -e "${YELLOW}⚙️ Configuring app settings...${NC}"
az webapp config appsettings set --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --settings \
    TELEGRAM_BOT_TOKEN="$BOT_TOKEN" \
    ADMIN_USER_IDS="$ADMIN_USER_ID" \
    ENVIRONMENT="production" \
    WEBSITE_RUN_FROM_PACKAGE="1" \
    SCM_DO_BUILD_DURING_DEPLOYMENT="1" \
    PYTHONPATH="/home/site/wwwroot" \
    --output table

echo -e "${GREEN}✅ App settings configured${NC}"

# Configure startup command
echo -e "${YELLOW}🔧 Configuring startup command...${NC}"
az webapp config set --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --startup-file "startup.sh" --output table || true

# Enable logging
echo -e "${YELLOW}📊 Enabling logging...${NC}"
az webapp log config --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --application-logging filesystem --level information --output table || true

# Check required files
echo -e "${YELLOW}📦 Checking required files...${NC}"
REQUIRED_FILES=(
    "bot.py"
    "azure_app.py"
    "requirements.txt"
    "startup.sh"
    "web.config"
    ".deployment"
    "your_analysis_module.py"
    "enhanced_indicators.py"
    "symbol_mapper.py"
    "upstox_fetcher.py"
    "perplexity_symbol_resolver.py"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}❌ Missing files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "   $file"
    done
    echo -e "${YELLOW}Please ensure all files are in the current directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required files found${NC}"

# Create deployment package
ZIP_FILE="trading-bot-deploy.zip"
echo -e "${YELLOW}📦 Creating deployment package...${NC}"

if [ -f "$ZIP_FILE" ]; then
    rm "$ZIP_FILE"
fi

if command -v zip &> /dev/null; then
    zip -r "$ZIP_FILE" "${REQUIRED_FILES[@]}"
else
    # Fallback using tar
    tar -czf "trading-bot-deploy.tar.gz" "${REQUIRED_FILES[@]}"
    ZIP_FILE="trading-bot-deploy.tar.gz"
fi

echo -e "${GREEN}✅ Deployment package created: $ZIP_FILE${NC}"

# Make startup script executable
chmod +x startup.sh

# Deploy to Azure
echo -e "${YELLOW}🚀 Deploying to Azure...${NC}"
if ! az webapp deployment source config-zip --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --src "$ZIP_FILE" --output table; then
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Deployment completed!${NC}"

# Wait for deployment
echo -e "${YELLOW}⏳ Waiting for app to start...${NC}"
sleep 30

# Set webhook
WEBHOOK_URL="https://${APP_NAME}.azurewebsites.net/webhook"
echo -e "${YELLOW}🔗 Setting Telegram webhook...${NC}"

if curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"$WEBHOOK_URL\"}" | grep -q '"ok":true'; then
    echo -e "${GREEN}✅ Webhook set successfully!${NC}"
    echo -e "   URL: $WEBHOOK_URL"
else
    echo -e "${RED}❌ Webhook setting failed${NC}"
fi

# Final status
echo
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETED!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${CYAN}App URL: https://${APP_NAME}.azurewebsites.net${NC}"
echo -e "${CYAN}Health Check: https://${APP_NAME}.azurewebsites.net/health${NC}"
echo -e "${CYAN}Webhook URL: ${WEBHOOK_URL}${NC}"
echo
echo -e "${GREEN}Your Enhanced Trading Bot is now live on Azure! 🚀${NC}"

# Cleanup
if [ -f "$ZIP_FILE" ]; then
    rm "$ZIP_FILE"
    echo -e "🧹 Cleaned up deployment package"
fi

echo
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo -e "1. Test your bot by sending /start message on Telegram"
echo -e "2. Check health status: https://${APP_NAME}.azurewebsites.net/health"
echo -e "3. Monitor logs in Azure Portal → App Service → Log stream"
echo -e "4. Consider enabling 'Always On' in Azure Portal for better performance"
echo

# Test webhook
echo -e "${YELLOW}🧪 Testing deployment...${NC}"
if python3 azure_deployment_helper.py &> /dev/null; then
    echo -e "${GREEN}✅ Deployment helper available for testing${NC}"
else
    echo -e "${YELLOW}⚠️ Run 'python3 azure_deployment_helper.py' to test your deployment${NC}"
fi

echo -e "${GREEN}Deployment script completed successfully! 🎉${NC}"