import os
from typing import Type

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# CrewAI 1.15.x + Groq compatibility patch
# ---------------------------------------------------------
# CrewAI can inject a `cache_breakpoint` field into messages.
# Groq does not support that field, so we disable the injection.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda message: message


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
    """Input schema for the custom web-search tool."""

    query: str = Field(
        ...,
        description="The web search query to run."
    )


# =========================================================
# CUSTOM FREE WEB SEARCH TOOL
# =========================================================

class FreeWebSearchTool(BaseTool):
    """Free web-search tool using DDGS."""

    name: str = "free_web_search"

    description: str = (
        "Search the public web for current information. "
        "Returns titles, snippets, and URLs. "
        "Use this tool when factual or current information is needed."
    )

    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        """Run a web search and return formatted results."""

        try:
            results = DDGS(timeout=10).text(
                query=query,
                max_results=5,
                backend="duckduckgo"
            )

        except Exception:
            try:
                results = DDGS(timeout=10).text(
                    query=query,
                    max_results=5,
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
                f"Result {index}\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)


# =========================================================
# LOAD GROQ API KEY
# =========================================================

def load_api_key():
    """
    Load GROQ_API_KEY from Streamlit secrets first,
    then from environment variables.
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# =========================================================
# CREATE GROQ LLM
# =========================================================

def create_groq_llm(api_key: str):
    """
    Create the Groq LLM used by CrewAI.

    The 'groq/' prefix tells CrewAI/LiteLLM which provider to use.
    The actual Groq model ID is openai/gpt-oss-120b.
    """

    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
    )


# =========================================================
# RUN THE RESEARCH AGENT
# =========================================================

def run_research(topic: str, api_key: str) -> str:
    """
    Create and run:
    - 1 Agent
    - 1 Task
    - 1 Crew
    """

    topic = topic.strip()

    if not topic:
        raise ValueError("Please enter a research topic.")

    if not api_key:
        raise ValueError("Groq API key is missing.")

    llm = create_groq_llm(api_key)

    search_tool = FreeWebSearchTool()

    # =====================================================
    # SINGLE RESEARCH AGENT
    # =====================================================

    researcher = Agent(
        role="AI Researcher and Report Writer",

        goal=(
            "Research the user's topic using current web information, "
            "compare multiple sources, identify important facts and evidence, "
            "and produce a clear and reliable research report."
        ),

        backstory=(
            "You are a careful research analyst. "
            "You use web search to investigate topics, compare sources, "
            "and avoid inventing facts, statistics, URLs, or references. "
            "If information is uncertain or conflicting, explain that clearly."
        ),

        llm=llm,

        tools=[search_tool],

        allow_delegation=False,

        verbose=False,

        # Lower than before to reduce Groq token-per-minute usage.
        max_iter=4,
    )

    # =====================================================
    # RESEARCH TASK
    # =====================================================

    research_task = Task(
        description=f"""
Research the following topic:

TOPIC:
{topic}

Instructions:

1. Use the free_web_search tool to find current information.
2. Perform a few focused searches rather than excessive searches.
3. Prefer authoritative and recent sources.
4. Compare multiple sources before making important claims.
5. Do NOT invent facts, statistics, studies, URLs, or references.
6. If sources disagree, explain the disagreement.
7. Keep the report useful but reasonably concise.

Write the report in Markdown using these sections:

# Title

## Executive Summary
Summarize the topic and the most important findings.

## Introduction / Background
Explain the topic and why it matters.

## Key Findings
List the most important findings.

## Detailed Discussion
Explain the findings in more detail.

## Important Facts and Evidence
Include useful facts, statistics, trends, or examples found during research.

## Conclusion
Summarize the overall findings.

## Sources / References
List the real sources used during research.

For each source include:
- Source title
- Website or organization
- Real URL

Never invent URLs or references.
""",

        expected_output=(
            "A concise but useful Markdown research report with "
            "real source URLs."
        ),

        agent=researcher,
    )

    # =====================================================
    # CREW
    # =====================================================

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False,
    )

    # =====================================================
    # RUN CREW
    # =====================================================

    result = crew.kickoff()

    if hasattr(result, "raw"):
        return result.raw

    return str(result)


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================

st.title("🔎 AI Research Agent")

st.write(
    """
Enter a research topic below.

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

For Streamlit Community Cloud, add this in **App settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
"""
    )


# =========================================================
# USER INPUT
# =========================================================

topic = st.text_area(
    "Research Topic",
    placeholder=(
        "Example: Research the impact of artificial intelligence "
        "on healthcare in 2026"
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
# RUN RESEARCH
# =========================================================

if research_button:

    if not topic.strip():
        st.error(
            "Please enter a research topic before clicking Research."
        )

    elif not groq_api_key:
        st.error(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud Secrets first."
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

            st.success("Research completed successfully.")

            st.divider()

            st.markdown(report)

            st.divider()

            st.download_button(
                label="⬇️ Download Research Report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as exc:

            error_text = str(exc)

            if "rate limit" in error_text.lower():
                st.error(
                    "Groq rate limit reached. "
                    "Wait a few seconds and try again."
                )

            else:
                st.error(
                    "The research agent encountered an error."
                )

            st.exception(exc)
