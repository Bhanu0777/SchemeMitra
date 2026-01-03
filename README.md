# 🏛️ SchemeMitra – AI Government Scheme Finder

A modern, AI-powered web application to help Indians discover government schemes they are eligible for.

## 🚀 Features

✨ **Modern UI/UX**
- Clean, card-based government-tech dashboard design
- Smooth CSS animations and hover effects
- Responsive and accessible interface
- Dark theme with gradient accents

🤖 **AI-Powered Intelligence**
- Azure OpenAI for intelligent scheme matching
- Azure Cognitive Services for text analysis
- Personalized eligibility explanations
- Match score calculation (%)

🔍 **Smart Discovery**
- Real-time search across schemes
- Category-based filtering (Farmers, Women, Youth, MSME, Education, Senior Citizens)
- Ministry and beneficiary type filters
- AI eligibility analysis

💾 **User Features**
- Session-based scheme bookmarking (no login needed)
- Basic scheme comparison
- Multi-language support (English + Hindi)
- Accessibility mode (high contrast, large text)
- Feedback mechanism

📊 **Data**
- Real Indian government schemes from official sources
- Verified sources (myscheme.gov.in, india.gov.in, pmindia.gov.in)
- Regular updates from government portals

⚖️ **Compliance**
- Clear disclaimer about non-official status
- No personal data storage
- No payments or login required
- GDPR and ethics-compliant

---

## 📋 System Requirements

- Python 3.9+
- Windows/Mac/Linux
- Internet connection for Azure services

---

## ⚙️ Installation & Setup

### Step 1: Clone/Download Project
```bash
# Navigate to your project folder
cd d:\c codes\MVP
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Azure Credentials

Create a `.env` file in the project root with your Azure credentials:

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_TEXTANALYTICS_KEY=your_text_analytics_key_here
AZURE_TEXTANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

**How to Get Azure Credentials:**

1. **Azure OpenAI Service:**
   - Go to https://portal.azure.com
   - Create/Select an Azure OpenAI resource
   - In Keys and Endpoint section, copy the API key and endpoint
   - Note your deployment name (usually "gpt-35-turbo" or "gpt-4")

2. **Azure Cognitive Services (Text Analytics):**
   - Create a Text Analytics resource in Azure Portal
   - Copy API key and endpoint from Keys and Endpoint section

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 🎯 How to Use

1. **Search**: Use the search bar to find schemes by name or keyword
2. **Browse Categories**: Click category icons (Farmers, Women, Youth, etc.)
3. **Filter**: Use dropdown filters for ministry and beneficiary type
4. **View Details**: Click "Why I'm Eligible" to see AI-generated explanations
5. **Match Score**: See percentage match based on your profile
6. **Bookmark**: Save schemes for later (stored in session)
7. **Language**: Switch to Hindi for scheme descriptions
8. **Accessibility**: Enable high contrast or large text mode

---

## 🏗️ Architecture

```
SchemeMitra/
├── app.py                  # Main Streamlit application
├── schemes.json           # Scheme database
├── .env                   # Azure credentials (create manually)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Key Components

**Frontend (Streamlit + CSS)**
- Custom HTML/CSS for modern UI
- Pure CSS animations (no external JS)
- Embedded styling for cards, buttons, and effects

**Backend (Python)**
- Streamlit framework for web interface
- Session state management for bookmarks
- Local data loading from JSON

**AI Services (Azure)**
- Azure OpenAI: Scheme matching and eligibility explanation
- Text Analytics: User input analysis and entity extraction

---

## 🤖 AI Integration Details

### How Matching Works

1. User enters their profile (age, category, skills)
2. Azure Text Analytics extracts key attributes
3. Azure OpenAI analyzes schemes
4. Match score calculated based on eligibility criteria
5. AI generates personalized "Why you're eligible" text

### Sample Prompts

- *"Find schemes for a 28-year-old farmer with 5 acres"*
- *"What government support is available for women entrepreneurs?"*
- *"I'm a senior citizen needing healthcare – help me find schemes"*

---

## 🎨 Design Features

### CSS Effects Implemented

✅ Color Fade - Smooth color transitions on hover
✅ Shadow Lift - Cards lift with enhanced shadows
✅ Underline Reveal - Animated underlines using ::after
✅ Icon Rotate - 360° rotation on category icons
✅ Card Zoom - Subtle scale transform
✅ Gradient Button Sweep - Button hover animation
✅ Glow Effect - Search bar focus glow
✅ Smooth Transitions - All animations use transitions and transforms

### Color Scheme

- Primary: Deep Blue (#1a3a52)
- Accent: Gradient Orange (#ff6b35 → #f7931e)
- Background: Light Gray (#f5f7fa)
- Text: Dark Gray (#333333)

---

## 🛡️ Disclaimer

⚠️ **IMPORTANT**: SchemeMitra is an independent application and is NOT an official government portal. 

- This platform provides guidance only
- Always verify information on official government portals
- Application creators are not responsible for inaccuracies
- Consult official government offices for official clarification

---

## 📞 Support & Feedback

The application includes a feedback button for user suggestions. Your feedback helps improve SchemeMitra!

---

## 🎓 For Imagine Cup & College Projects

**Architecture Highlights:**
- Cloud-native design using Azure services
- Scalable AI integration
- Modern web framework (Streamlit)
- Production-ready code quality
- Beginner-friendly comments
- Real-world problem solving

This MVP demonstrates:
- Full-stack web development capabilities
- Cloud AI service integration
- UX/UI design thinking
- Real government data utilization
- Social impact technology

---

## 🚀 Future Enhancements

- User authentication and saved preferences
- Email/SMS notifications for new schemes
- Mobile app version
- Scheme comparison matrix
- Application form auto-filling
- Community forums and success stories
- Admin dashboard for data updates
- Multi-language expansion (10+ languages)

---

## 📄 License

This project is open-source and intended for educational and non-commercial use.

---

## 👥 Contributors

Built with ❤️ for the Imagine Cup and Indian communities

---

**Last Updated**: January 3, 2026

**Status**: ✅ Production Ready for MVP
