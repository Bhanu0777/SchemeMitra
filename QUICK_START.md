# 🏛️ SchemeMitra - QUICK REFERENCE GUIDE

## 🚀 TL;DR (5-Minute Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file (copy from .env.example)
copy .env.example .env

# 3. Edit .env with your Azure credentials
# (Get them from Azure Portal)

# 4. Run the app
streamlit run app.py

# 5. Open browser → http://localhost:8501
```

---

## 📋 Project Structure

```
SchemeMitra/
├── app.py                    ← Main Streamlit application
├── schemes.json              ← Real government schemes data
├── requirements.txt          ← Python dependencies
├── .env.example              ← Template for credentials
├── README.md                 ← Full documentation
├── SETUP.py                  ← Automated setup script
└── QUICK_START.md           ← This file
```

---

## 🔑 Getting Azure Credentials (2 Minutes)

### Step 1: Create Azure OpenAI Resource
1. Go to https://portal.azure.com
2. Click "Create a resource" → Search "Azure OpenAI"
3. Click Create
4. Fill details (free tier available):
   - Resource name: `schemitra-openai`
   - Region: Any region
   - Pricing tier: Standard
5. Click Review + Create → Create
6. Wait for deployment (2-3 minutes)
7. Go to resource → Click "Keys and Endpoint"
8. Copy:
   - **Key 1** → `AZURE_OPENAI_API_KEY`
   - **Endpoint** → `AZURE_OPENAI_ENDPOINT`

### Step 2: Deploy a Model
1. In the OpenAI resource, click "Model deployments" or "Go to Azure OpenAI Studio"
2. Create new deployment:
   - Model: gpt-3.5-turbo
   - Deployment name: `gpt-35-turbo`
3. Note the deployment name
4. In .env: `AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo`

### Step 3: Create Text Analytics Resource
1. Go back to Azure Portal
2. Create a resource → Search "Text Analytics"
3. Click Create
4. Fill details:
   - Resource name: `schemitra-text`
   - Pricing tier: Free (limit: 5000 records/month)
5. Click Create
6. Go to resource → Click "Keys and Endpoint"
7. Copy:
   - **Key 1** → `AZURE_TEXTANALYTICS_KEY`
   - **Endpoint** → `AZURE_TEXTANALYTICS_ENDPOINT`

---

## ⚙️ Environment Variables (.env)

Create a `.env` file in the project root:

```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_TEXTANALYTICS_KEY=your-key-here
AZURE_TEXTANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

⚠️ **IMPORTANT**: Never commit `.env` to GitHub!

---

## 🎯 Key Features

### Search & Discovery
- **Search Box**: Find schemes by name, keyword, ministry
- **Category Filters**: Farmers, Women, Youth, MSME, Education, Senior Citizens
- **Smart Filters**: By ministry, beneficiary type, category

### AI Capabilities
- **Eligibility Matching**: Azure OpenAI matches user profile with schemes
- **Match Score**: Percentage-based eligibility indicator (0-100%)
- **Personalized Explanations**: AI-generated "Why you're eligible" text
- **Text Analysis**: Azure Text Analytics extracts key info from searches

### User Features
- **Bookmarking**: Save schemes (stored in browser session)
- **Language Toggle**: English & Hindi support
- **Accessibility Mode**: Large text + high contrast option
- **Feedback System**: User satisfaction tracking
- **Official Links**: Direct to government scheme portals

### UI/UX Features
- ✨ **Smooth Animations**: CSS-only (no JavaScript)
- 🎨 **Modern Design**: Government-tech dashboard style
- 📱 **Responsive**: Works on desktop, tablet, mobile
- ⚡ **Fast**: Instant filtering and search
- ♿ **Accessible**: WCAG-compliant

---

## 💡 How AI Works

### 1. User Input Analysis
```
User enters: "I'm a 25-year-old farmer"
↓
Azure Text Analytics extracts: age=25, category=farmer
```

### 2. Scheme Matching
```
System finds matching schemes: PM Kisan, NRLM, Housing
↓
Azure OpenAI analyzes user profile against each scheme
↓
Generates match score (0-100%)
```

### 3. Explanation Generation
```
"Why You're Eligible" section
↓
Azure OpenAI prompt:
  "User is 25-year-old farmer
   Scheme: PM Kisan
   Generate simple 2-3 sentence explanation"
↓
Returns personalized explanation
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### "Azure credentials not found" or "API key error"
1. Check `.env` file exists in project root
2. Check `.env` has correct format (no quotes around values)
3. Verify credentials from Azure Portal
4. Make sure Azure resources are created and deployed

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### Slow responses from AI
- Azure OpenAI might be throttled (rate limits)
- Wait 30 seconds and try again
- Check Azure Portal usage/quotas

### Schemes not loading
- Check `schemes.json` exists and is valid JSON
- Open schemes.json and verify it's not corrupted
- Run: `python -m json.tool schemes.json`

---

## 📊 Real Schemes Data

All schemes in `schemes.json` are from official sources:
- **Source 1**: https://myscheme.gov.in
- **Source 2**: https://india.gov.in
- **Source 3**: https://pmindia.gov.in
- **Source 4**: Ministry official websites

Each scheme includes:
- ✅ Official name
- ✅ Issuing ministry
- ✅ Beneficiary type
- ✅ Benefits offered
- ✅ Official source URL
- ✅ Description

---

## 🎨 Customization

### Add More Schemes
Edit `schemes.json`:
```json
{
  "schemes": [
    {
      "id": "custom001",
      "name": "Your Scheme Name",
      "ministry": "Ministry Name",
      "category": "Farmers|Women|Youth|MSME|Education|Senior Citizens",
      "beneficiary": "Description of who can apply",
      "benefit": "What they get",
      "status": "Active",
      "source_url": "https://...",
      "source_name": "Portal Name",
      "description": "Detailed description"
    }
  ]
}
```

### Change Colors
Edit `app.py`, search for "GLOBAL STYLES" section:
```css
:root {
    --primary-color: #1a3a52;      /* Change this */
    --accent-color: #ff6b35;       /* And this */
    --success-color: #10b981;      /* And this */
    ...
}
```

### Disable AI Features
Comment out these functions in `app.py`:
- `call_azure_openai()`
- `analyze_text_azure()`
- `generate_eligibility_explanation()`

---

## 📱 Running on Different Ports

```bash
# Default (8501)
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Public IP (for network access)
streamlit run app.py --server.address 0.0.0.0
```

---

## 🔒 Security Notes

✅ **What's Secure:**
- No login/authentication system
- No personal data storage
- No database connections
- Session-based bookmarks only (cleared on refresh)
- All API calls use Azure secure endpoints

⚠️ **Keep Safe:**
- Never commit `.env` to Git
- Don't share your Azure API keys
- Use Azure role-based access control
- Monitor Azure usage/costs

---

## 📈 For Imagine Cup Judges

### Innovation Points
- ✨ AI-powered eligibility matching
- 🤖 Azure OpenAI + Text Analytics integration
- 💡 Real government scheme data
- ♿ Accessibility-first design
- 🌍 Multi-language support

### Technical Excellence
- 📦 Clean, modular code (600+ lines with comments)
- 🎨 Modern UI with pure CSS animations
- ⚡ Fast, responsive interface
- 🔐 Secure credential handling
- 📱 Mobile-responsive design

### Social Impact
- 🇮🇳 Solves real problem for Indians
- 📊 Data from official government portals
- 🎓 Educational resource
- ♿ Accessible to all users
- 🚀 Scalable architecture

---

## 📞 Support

### For Setup Issues
1. Check README.md (detailed documentation)
2. Run `SETUP.py` (automated setup wizard)
3. Review `.env.example` (credentials template)

### For Azure Issues
- https://portal.azure.com (manage resources)
- https://learn.microsoft.com/en-us/azure/ (official docs)
- Azure support tickets

### For Code Issues
- Check syntax with: `python -m py_compile app.py`
- Check JSON with: `python -m json.tool schemes.json`
- Run in debug mode: `streamlit run app.py --logger.level=debug`

---

## 🚀 Deployment (Advanced)

### Deploy to Azure App Service
```bash
# Install Azure CLI
# https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create -n schemitra-rg -l eastus

# Create App Service Plan
az appservice plan create -n schemitra-plan -g schemitra-rg --sku FREE

# Create Web App
az webapp create -n schemitra-app -g schemitra-rg --plan schemitra-plan

# Deploy
az webapp up -n schemitra-app -g schemitra-rg
```

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repo
4. Add `.env` secrets in dashboard
5. Deploy!

---

## 📝 License

Open source | Educational use | Non-commercial

---

**Last Updated**: January 3, 2026
**Status**: ✅ Production Ready for MVP

🎉 **Ready to discover government schemes?** Run `streamlit run app.py` now!
