# AI Research Agent

A beginner-friendly single-agent research application built with CrewAI, Groq, DDGS, and Streamlit.

## Features

- One CrewAI research agent
- One research task
- Free web search through DDGS
- Groq `openai/gpt-oss-120b`
- Structured Markdown research reports
- Streamlit interface
- GitHub- and Streamlit-Cloud-ready
- API key kept outside source code

## Architecture

```text
User Topic
    ↓
Streamlit UI
    ↓
CrewAI Crew
    ↓
Single Research Agent
    ↓
DDGS Web Search Tool
    ↓
Groq openai/gpt-oss-120b
    ↓
Structured Research Report
```

## Requirements

Use Python 3.10, 3.11, 3.12, or 3.13. Python 3.11 is a good beginner-friendly choice.

## Installation

Clone the repository and enter the project folder:

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-agent.git
cd ai-research-agent
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Groq API setup

Create a Groq API key in the Groq console.

For local Streamlit development, create:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your_real_groq_api_key_here"
```

Do not commit this file. It is already listed in `.gitignore`.

You may alternatively set an environment variable named `GROQ_API_KEY`.

## Run locally

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal, normally:

```text
http://localhost:8501
```

Example topic:

```text
Research the impact of artificial intelligence on healthcare
```

## Why Groq is configured this way

The project connects CrewAI directly to Groq's OpenAI-compatible endpoint rather than relying on an older LiteLLM-specific Groq setup.

The CrewAI LLM configuration uses:

```python
LLM(
    model="openai/gpt-oss-120b",
    custom_openai=True,
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)
```

This keeps the model ID expected by Groq while using CrewAI's OpenAI-compatible provider path.

## GitHub upload

Create a new empty GitHub repository named `ai-research-agent`, then run:

```bash
git init
git add .
git status
git commit -m "Initial AI research agent"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-research-agent.git
git push -u origin main
```

Before committing, verify that `.streamlit/secrets.toml` does not appear in `git status`.

## Streamlit Community Cloud deployment

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Choose **Create app**.
4. Select your `ai-research-agent` repository.
5. Select the `main` branch.
6. Set the main file to `app.py`.
7. Open the app's **Secrets** settings.
8. Add:

```toml
GROQ_API_KEY = "your_real_groq_api_key_here"
```

9. Deploy the app.
10. Enter a research topic and test the result.

## Google Colab API-key test

For a temporary Colab test, you can use:

```python
import os
from getpass import getpass

os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API key: ")
```

The input is hidden while you type. Do not write the real key directly into a public notebook cell.

## Troubleshooting

### GROQ_API_KEY is missing

Create `.streamlit/secrets.toml` locally or add the key to Streamlit Cloud's Secrets settings.

### Invalid model

Verify the model is still available in Groq and that the model string is exactly:

```text
openai/gpt-oss-120b
```

### CrewAI provider error

Make sure `research_agent.py` uses `custom_openai=True` and the Groq base URL:

```text
https://api.groq.com/openai/v1
```

Then reinstall dependencies:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### DDGS search fails

Free public search backends can occasionally rate-limit or fail. The code first tries DuckDuckGo and then falls back to DDGS's automatic backend selection.

Try again later if the free search provider is temporarily blocked.

### Import errors

Activate the virtual environment and reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Streamlit secrets error

Check that the local file is exactly:

```text
.streamlit/secrets.toml
```

and contains valid TOML:

```toml
GROQ_API_KEY = "your_key"
```

### Rate limits

Groq and free search providers may enforce rate limits. Wait and retry, reduce repeated runs, or check the relevant provider dashboard.

## Security

Never commit your real API key to GitHub.

If a secret is accidentally committed, remove it from the repository and revoke/rotate the key immediately in Groq.
