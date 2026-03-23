import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# --- 1. HELPER FUNCTIONS & CONFIG ---
def wrap_lime_html(html_data):
    """Wraps LIME HTML in a white container to fix dark mode visibility issues."""
    return f"""
    <div style="background-color: white; color: black !important; padding: 20px; border-radius: 10px; font-family: sans-serif; border: 1px solid #ddd;">
        <style> * {{ color: black !important; }} </style>
        {html_data}
    </div>
    """

st.set_page_config(page_title="AI vs Human Detector Pro", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
# Using session state keys to prevent the "reverting text" bug
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analyzed_text' not in st.session_state:
    st.session_state.analyzed_text = ""

# --- 3. SIDEBAR WARNINGS (The Research Audit Layer) ---
with st.sidebar:
    st.header("🚦 Model Reliability Notes")
    st.warning("""
    **Short Text Warning:** Predictions on texts under 50 words are statistically less reliable. NLP models require "textural" patterns to detect AI signatures effectively.
    """)
    st.info("""
    **LSTM Overconfidence:** Sequence models (LSTM/GRU) often exhibit 'Sigmoid Saturation', leading to 95%+ confidence scores even on ambiguous data. Treat these as rankings, not absolute certainties.
    """)
    st.error("""
    **The Generalization Gap:** Models trained on static datasets may struggle with evolving LLMs (like GPT-4). This app serves as an audit tool to identify these blind spots.
    """)

# --- 4. MAIN UI ---
st.title("🕵️‍♂️ Multi-Paradigm AI Detector")
st.markdown("Compare baseline ML, Deep Learning, and SOTA Transformers with XAI Explainability.")

# We use the key 'current_input' to manage widget state internally
user_text = st.text_area("Paste your text here:", height=200, key="current_input")

# --- 5. ANALYZE BUTTON LOGIC ---
if st.button("Analyze Text", type="primary"):
    # Pull text directly from the session state key to avoid the "reverting" bug
    input_to_analyze = st.session_state.current_input
    
    if input_to_analyze.strip():
        with st.spinner("Analyzing across all paradigms..."):
            try:
                # Clear old XAI results from previous runs
                for m_id in ['lr', 'lstm', 'bert']:
                    if f"lime_{m_id}" in st.session_state:
                        del st.session_state[f"lime_{m_id}"]
                
                response = requests.post("http://127.0.0.1:8000/predict_all", json={"text": input_to_analyze})
                
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.session_state.analyzed_text = input_to_analyze
                else:
                    st.error("Backend Error: Ensure main.py is running on port 8000.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        st.error("Please enter some text first.")

# --- 6. DISPLAY RESULTS ---
if st.session_state.results:
    results = st.session_state.results
    original_text = st.session_state.analyzed_text
    
    st.divider()
    
    # Conditional Warning for Short Text
    if len(original_text.split()) < 30:
        st.warning("⚠️ **Low Word Count:** Model confidence may be artificially inflated for short sentences.")

    col1, col2, col3 = st.columns(3)

    def show_model_card(column, title, m_id, data):
        with column:
            st.subheader(title)
            color = "red" if data['label'] == "AI" else "green"
            st.markdown(f"### Status: :{color}[{data['label']}]")
            st.metric("Confidence", f"{data['confidence']:.2%}")
            st.progress(data['confidence'])
            
            # On-Demand XAI
            if st.button(f"🔍 Explain {m_id.upper()}", key=f"expl_{m_id}"):
                with st.spinner(f"Computing LIME for {title}..."):
                    params = {"text": original_text, "model_type": m_id}
                    exp_res = requests.get("http://127.0.0.1:8000/explain", params=params)
                    if exp_res.status_code == 200:
                        st.session_state[f"lime_{m_id}"] = exp_res.text
                    else:
                        st.error("Explanation failed.")

            # Persistent XAI Display
            if f"lime_{m_id}" in st.session_state:
                st.write("**Feature Importance:**")
                styled_html = wrap_lime_html(st.session_state[f"lime_{m_id}"])
                components.html(styled_html, height=450, scrolling=True)

    show_model_card(col1, "Logistic Regression (ML)", "lr", results['lr'])
    show_model_card(col2, "LSTM (Deep Learning)", "lstm", results['lstm'])
    show_model_card(col3, "BERT (Transformer)", "bert", results['bert'])

# --- 7. SQL LOGGING HISTORY ---
st.divider()
with st.expander("📜 View Prediction Logs (SQL Database)"):
    if st.button("Refresh History"):
        try:
            h_res = requests.get("http://127.0.0.1:8000/history")
            if h_res.status_code == 200:
                data = h_res.json()
                df = pd.DataFrame(data, columns=[
                    "ID", "Text Snippet", "LR Label", "LR Conf", 
                    "LSTM Label", "LSTM Conf", "BERT Label", "BERT Conf", "Timestamp"
                ])
                st.dataframe(df, width='stretch')
            else:
                st.error("Could not fetch history.")
        except:
            st.error("Database connection failed.")