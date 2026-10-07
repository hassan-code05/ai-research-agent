```python
import os
from typing import Type

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# SEARCH TOOL INPUT SCHEMA
# =========================================================

class WebSearchInput(BaseModel):
    """
    Defines what input our custom CrewAI search tool expects.
    """

    query: str = Field(
        ...,
        description="The search query to look up on the web."
    )


# =========================================================
# CUSTOM FREE WEB SEARCH TOOL
# =========================================================

class FreeWebSearchTool(BaseTool):
    """
    A simple CrewAI tool that performs free web searches using DDGS.
    """

    name: str = "free_web_search"

    description: str = (
        "Search the public web for current information. "
        "Returns titles, snippets, and URLs. "
        "Use this tool whenever you need factual or current information."
    )

    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """
        Runs the web search.
        """

        try:
            # First try DuckDuckGo directly
            results = DDGS(timeout=10).text(
                query=query,
                max_results=6,
                backend="duckduckgo"
            )

        except Exception:
            try:
                # If DuckDuckGo fails, DDGS can try another free backend
                results = DDGS(timeout=10).text(
                    query=query,
                    max_results=6,
                    backend="auto"
                )

            except Exception as exc:
                return (
                    f"Web search failed for query '{query}'.\n"
                    f"Error: {exc}"
                )

        if not results:
            return f"No search results were found for: {query}"

        formatted_results = []

        for index, item in enumerate(results, start=1):
            title = item.get("title", "Untitled result")
            snippet = item.get("body", "No description available")
            url = item.get("href", "No URL available")

            formatted_results.append(
                f"""
Result {index}
Title: {title}
Snippet: {snippet}
URL: {url}
"""
            )

        return "\n".join(formatted_results)


# =========================================================
# LOAD GROQ API KEY
# =========================================================

def load_api_key():
    """
    Tries to load GROQ_API_KEY from:

    1. Streamlit secrets
    2. Environment variables

    Returns None if no key is found.
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# =========================================================
# BUILD GROQ LLM
# =========================================================

def create_groq_llm(api_key: str):
    """
    Creates the LLM used by CrewAI.

    Groq provides an OpenAI-compatible API,
    so CrewAI can connect to Groq through this endpoint.
    """

    return LLM(
        model="openai/gpt-oss-120b",

        custom_openai=True,

        base_url="https://api.groq.com/openai/v1",

        api_key=api_key,

        temperature=0.2,

        max_tokens=7000,
    )


# =========================================================
# RUN THE RESEARCH AGENT
# =========================================================

def run_research(topic: str, api_key: str) -> str:
    """
    Creates and runs:

    1 Agent
    1 Task
    1 Crew

    Then returns the final research report.
    """

    topic = topic.strip()

    if not topic:
        raise ValueError("Please enter a research topic.")

    if not api_key:
        raise ValueError(
            "Groq API key is missing."
        )

    # Create our Groq LLM
    llm = create_groq_llm(api_key)

    # Create our free web search tool
    search_tool = FreeWebSearchTool()

    # =====================================================
    # CREATE THE SINGLE RESEARCH AGENT
    # =====================================================

    researcher = Agent(

        role="Senior AI Researcher and Report Writer",

        goal=(
            "Research the user's topic using current web information, "
            "compare multiple sources, identify important facts and evidence, "
            "and produce a clear and reliable research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You investigate topics carefully using web search. "
            "You compare information from multiple sources before making "
            "strong claims. You never invent URLs, sources, statistics, "
            "or facts. If information is uncertain or conflicting, "
            "you clearly mention that."
        ),

        llm=llm,

        tools=[search_tool],

        allow_delegation=False,

        verbose=False,

        max_iter=8,
    )

    # =====================================================
    # CREATE THE RESEARCH TASK
    # =====================================================

    research_task = Task(

        description=f"""
Research the following topic thoroughly:

TOPIC:
{topic}

Your instructions:

1. Use the free_web_search tool multiple times.
2. Use focused search queries rather than only one broad search.
3. Prefer current and authoritative sources.
4. Compare multiple sources before making important claims.
5. Do NOT invent facts, statistics, studies, URLs, or references.
6. If sources disagree, clearly explain the disagreement.
7. Write a professional Markdown research report.

The report must contain:

# Title

## Executive Summary

Provide a concise summary of the topic and the most important findings.

## Introduction / Background

Explain the topic and why it is important.

## Key Findings

Present the most important findings clearly.

## Detailed Discussion

Explain the topic in detail using information from your research.

## Important Facts and Evidence

Include useful statistics, evidence, trends, findings, or examples.

## Conclusion

Summarize the overall findings.

## Sources / References

List the real sources used during research.

For every source, include:

- Source title
- Website or organization
- Real URL

Never create fake URLs or fake references.
""",

        expected_output=(
            "A complete Markdown research report containing a title, "
            "executive summary, background, key findings, detailed discussion, "
            "evidence, conclusion, and real source URLs."
        ),

        agent=researcher,
    )

    # =====================================================
    # CREATE THE CREW
    # =====================================================

    crew = Crew(

        agents=[researcher],

        tasks=[research_task],

        process=Process.sequential,

        verbose=False,
    )

    # =====================================================
    # RUN THE CREW
    # =====================================================

    result = crew.kickoff()

    # CrewAI usually provides the final answer inside .raw
    if hasattr(result, "raw"):
        return result.raw

    return str(result)


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================

st.title("🔎 AI Research Agent")

st.write(
    """
Enter any research topic below.

A single CrewAI research agent will:

- Search the web
- Analyze multiple sources
- Compare information
- Generate a structured research report
- Provide source links
"""
)


# =========================================================
# CHECK API KEY
# =========================================================

groq_api_key = load_api_key()


if not groq_api_key:

    st.warning(
        """
Groq API key not found.

For local testing, create:

`.streamlit/secrets.toml`

and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

For Streamlit Community Cloud, add the same key
inside your application's **Secrets** settings.
"""
    )


# =========================================================
# USER INPUT
# =========================================================

topic = st.text_area(

    "Research Topic",

    placeholder=(
        "Example: Research the impact of artificial intelligence "
        "on healthcare"
    ),

    height=140,
)


# =========================================================
# RESEARCH BUTTON
# =========================================================

research_button = st.button(

    "🔍 Research",

    type="primary",

    use_container_width=True,
)


# =========================================================
# RUN RESEARCH WHEN BUTTON IS CLICKED
# =========================================================

if research_button:

    # Check topic
    if not topic.strip():

        st.error(
            "Please enter a research topic before clicking Research."
        )

    # Check API key
    elif not groq_api_key:

        st.error(
            """
GROQ_API_KEY is missing.

Please add your Groq API key first.
"""
        )

    else:

        try:

            with st.spinner(
                "Researching the web and generating your report..."
            ):

                report = run_research(
                    topic=topic,
                    api_key=groq_api_key
                )

            st.success(
                "Research completed successfully."
            )

            st.divider()

            # =================================================
            # DISPLAY REPORT
            # =================================================

            st.markdown(report)

            st.divider()

            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            st.download_button(

                label="⬇️ Download Research Report",

                data=report,

                file_name="research_report.md",

                mime="text/markdown",

                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                "The research agent encountered an error."
            )

            st.exception(exc)
```

Now your project is much simpler:

```text
ai-research-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

Your `requirements.txt` remains:

```txt
crewai[openai]==1.15.22
streamlit==1.65.0
ddgs==9.16.0
```

Locally, create:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your_real_groq_key"
```

Then run:

```bash
pip install -r requirements.txt
```

and:

```bash
streamlit run app.py
```

On **Streamlit Community Cloud**, you only select:

```text
app.py
```

as the **Main file path**. Streamlit still automatically reads `requirements.txt`, so you do need to upload that too.

This one-file version is the one I'd recommend for your project right now. Once it works, you can split the agent code into separate files later if the project grows.
