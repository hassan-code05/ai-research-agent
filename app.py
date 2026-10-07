
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
