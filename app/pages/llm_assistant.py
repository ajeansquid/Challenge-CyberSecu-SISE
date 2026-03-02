# -*- coding: utf-8 -*-
"""
LLM Assistant Page
------------------
Three modes:
  1. Data Analysis Interpreter  – summarise / explain loaded data & model results
  2. Dataset Q&A Chat           – ask natural-language questions, LLM writes pandas code
  3. Raw Log Explainer          – paste firewall / syslog lines, get plain-language explanation
"""

import os
import io
import textwrap
import traceback
from typing import Optional

import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app.state import get_state

# ---------------------------------------------------------------------------
# Mistral client (lazy-init)
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env at repo root

_MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
_MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


def _get_client():
    """Return a MistralClient, or None if the key is missing."""
    if not _MISTRAL_API_KEY or _MISTRAL_API_KEY == "your-key-here":
        return None
    from mistralai import Mistral
    return Mistral(api_key=_MISTRAL_API_KEY)


def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    """Send a single system+user prompt to Mistral and return the text."""
    client = _get_client()
    if client is None:
        return "⚠️ Mistral API key not configured. Add it to `.env`."
    response = client.chat.complete(
        model=_MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    """Render the LLM assistant page."""
    state = get_state()

    st.title("🤖 LLM Assistant")

    if not _MISTRAL_API_KEY or _MISTRAL_API_KEY == "your-key-here":
        st.error(
            "**Mistral API key not found.**  \n"
            "Create a `.env` file at the project root with:  \n"
            "```\nMISTRAL_API_KEY=your-key-here\n```"
        )

    tab_interpret, tab_qa, tab_logs = st.tabs([
        "📊 Data Interpreter",
        "💬 Dataset Q&A",
        "📝 Log Explainer",
    ])

    with tab_interpret:
        _render_interpreter(state)

    with tab_qa:
        _render_qa(state)

    with tab_logs:
        _render_log_explainer()


# ---------------------------------------------------------------------------
# Tab 1 – Data Analysis Interpreter
# ---------------------------------------------------------------------------

def _render_interpreter(state):
    """Summarise / interpret whatever data is currently loaded."""
    st.header("Data Analysis Interpreter")
    st.markdown("Get a plain-language summary and interpretation of loaded data or model results.")

    df, source = _best_dataframe(state)
    if df is None:
        st.info("Load some data first (upload, generate features, or train a model).")
        return

    st.caption(f"Using **{source}** ({len(df)} rows × {len(df.columns)} cols)")

    focus = st.selectbox(
        "What should the LLM focus on?",
        [
            "General summary & key insights",
            "Anomaly / outlier highlights",
            "Feature importance interpretation",
            "Security risk assessment",
            "Custom prompt…",
        ],
        key="interp_focus",
    )

    custom_prompt = ""
    if focus == "Custom prompt…":
        custom_prompt = st.text_area("Your prompt", key="interp_custom")

    if st.button("Interpret", type="primary", key="btn_interpret"):
        with st.spinner("Thinking…"):
            context = _dataframe_summary(df)
            system = textwrap.dedent("""\
                You are a senior cybersecurity data analyst.
                You are given summary statistics of a dataset derived from firewall logs.
                The 11 course-standard features per source IP are:
                nombre (total accesses), cnbripdst (unique dest IPs), cnportdst (unique dest ports),
                permit, inf1024permit, sup1024permit, adminpermit,
                deny, inf1024deny, sup1024deny, admindeny.
                'risque' is the binary label: positif = threat, negatif = benign.
                Provide clear, actionable analysis in Markdown. Use bullet points.
            """)
            if focus == "Custom prompt…":
                user_msg = f"Dataset context:\n{context}\n\nUser request: {custom_prompt}"
            else:
                user_msg = f"Dataset context:\n{context}\n\nFocus: {focus}"
            answer = _chat(system, user_msg)
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Tab 2 – Dataset Q&A Chat
# ---------------------------------------------------------------------------

_QA_SYSTEM = textwrap.dedent("""\
    You are a data analysis assistant embedded in a Streamlit cybersecurity toolkit.
    The user has a pandas DataFrame called `df`. They will ask natural-language
    questions. Respond with TWO sections:

    1. **Code** – a short Python/pandas snippet (use only pandas/numpy) that
       answers the question. The code MUST assign its result to a variable called
       `result`. Do NOT use print(). Wrap the code in a ```python fenced block.
    2. **Explanation** – a brief interpretation of what the code does and what
       the user should look for in the output.

    The DataFrame has these columns: {columns}
    And these dtypes:
    {dtypes}
    First 3 rows:
    {head}
""")


def _render_qa(state):
    """Interactive Q&A: user asks question → LLM writes pandas code → we run it."""
    st.header("Dataset Q&A")
    st.markdown("Ask questions about your data in plain English. The LLM writes pandas code and runs it.")

    df, source = _best_dataframe(state)
    if df is None:
        st.info("Load some data first.")
        return

    st.caption(f"Querying **{source}** ({len(df)} rows)")

    # Chat history
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Display past messages
    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New question
    question = st.chat_input("Ask about your data…")
    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer…"):
                system = _QA_SYSTEM.format(
                    columns=list(df.columns),
                    dtypes=df.dtypes.to_string(),
                    head=df.head(3).to_string(),
                )
                raw_answer = _chat(system, question, temperature=0.2)

            st.markdown(raw_answer)

            # Try to extract and execute code
            code = _extract_code(raw_answer)
            if code:
                with st.expander("▶ Run generated code", expanded=True):
                    try:
                        local_ns = {"df": df, "pd": pd, "np": np}
                        exec(code, {}, local_ns)  # noqa: S102
                        result = local_ns.get("result")
                        if result is not None:
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result, width='stretch')
                            elif isinstance(result, pd.Series):
                                st.dataframe(result.to_frame(), width='stretch')
                            else:
                                st.write(result)
                        else:
                            st.info("Code ran but did not assign `result`.")
                    except Exception:
                        st.error(f"Execution error:\n```\n{traceback.format_exc()}\n```")

        st.session_state.qa_messages.append({"role": "assistant", "content": raw_answer})

    # Clear chat button
    if st.session_state.qa_messages:
        if st.button("Clear chat", key="btn_clear_qa"):
            st.session_state.qa_messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 – Raw Log Explainer
# ---------------------------------------------------------------------------

_LOG_SYSTEM = textwrap.dedent("""\
    You are a cybersecurity log analyst. The user will paste raw firewall,
    syslog, or IDS log lines. For each line or group of lines:
    1. Identify the log format (e.g. iptables, Cisco ASA, syslog, Suricata).
    2. Parse key fields: timestamp, source IP, dest IP, port, protocol, action.
    3. Explain in plain language what happened.
    4. Flag anything suspicious (port scans, brute-force patterns, lateral
       movement, data exfiltration indicators, etc.).
    5. Suggest follow-up investigation steps if warranted.
    Use Markdown formatting with headers and bullet points.
""")


def _render_log_explainer():
    """Paste raw log lines and get a plain-language explanation."""
    st.header("Raw Log Explainer")
    st.markdown("Paste firewall / syslog / IDS log lines and get a security-focused explanation.")

    logs = st.text_area(
        "Paste log lines here",
        height=250,
        placeholder="Mar  1 12:34:56 fw01 kernel: IPTABLES-DROP IN=eth0 …",
        key="log_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        depth = st.selectbox("Detail level", ["Brief", "Detailed", "Full forensic"], key="log_depth")

    if st.button("Explain Logs", type="primary", key="btn_logs"):
        if not logs.strip():
            st.warning("Paste some log lines first.")
            return
        with st.spinner("Analysing logs…"):
            user_msg = f"Detail level: {depth}\n\nLogs:\n```\n{logs}\n```"
            answer = _chat(_LOG_SYSTEM, user_msg, temperature=0.2)
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _best_dataframe(state):
    """Return (df, label) for the best available data, or (None, None)."""
    if state.has_predictions():
        return state.predictions, "predictions"
    if state.has_features():
        return state.features_data, "features"
    if state.has_labeled_data():
        return state.labeled_data, "labeled data"
    if state.has_raw_data():
        return state.raw_data, "raw data"
    return None, None


def _dataframe_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Build a compact text summary of a DataFrame for the LLM context window."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape}\n")
    buf.write(f"Columns: {list(df.columns)}\n")
    buf.write(f"Dtypes:\n{df.dtypes.to_string()}\n\n")
    buf.write("Describe (numeric):\n")
    buf.write(df.describe().to_string())
    buf.write("\n\nHead:\n")
    buf.write(df.head(max_rows).to_string())

    # If classification labels present, add class balance
    for col in ("risque", "prediction"):
        if col in df.columns:
            buf.write(f"\n\nValue counts ({col}):\n")
            buf.write(df[col].value_counts().to_string())

    return buf.getvalue()


def _extract_code(text: str) -> Optional[str]:
    """Extract the first ```python … ``` fenced block from LLM output."""
    marker = "```python"
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = text.find("```", start)
    if end == -1:
        return None
    return text[start:end].strip()
