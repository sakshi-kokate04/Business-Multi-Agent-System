# Business Multi-Agent System

A CrewAI-powered multi-agent system for business analysis using 4 specialized AI agents.

## Agents
- 📊 Market Research Agent
- 💼 Sales Strategy Agent  
- 👥 Customer Analytics Agent
- 📝 Content Writer Agent

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create .env file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

3. Run the app:
```bash
streamlit run business_agents_crewai.py
```

## Deployment to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Deploy your repo
4. Add OPENAI_API_KEY in Streamlit secrets

## Usage

1. Select task type
2. Describe your business need
3. Click "Run Multi-Agent Analysis"
4. Review results from all agents
```

### 4. **.gitignore** (Optional but recommended)
```
.env
__pycache__/
*.pyc
.streamlit/secrets.toml