import streamlit as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import re
import sqlite3
from datetime import datetime
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

# --- 2. SQL DATABASE INITIALIZATION ---
DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            lr_label TEXT, lr_conf REAL,
            lstm_label TEXT, lstm_conf REAL,
            bert_label TEXT, bert_conf REAL,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_to_sql(text, results):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_logs 
        (input_text, lr_label, lr_conf, lstm_label, lstm_conf, bert_label, bert_conf, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        text[:500],
        results['lr']['label'], results['lr']['confidence'],
        results['lstm']['label'], results['lstm']['confidence'],
        results['bert']['label'], results['bert']['confidence'],
        datetime.now()
    ))
    conn.commit()
    conn.close()

# --- 3. MODEL INITIALIZATION (CLOUD MONOLITH) ---
device = torch.device("cpu")
explainer = LimeTextExplainer(class_names=['Human', 'AI'])

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.sigmoid(self.linear(hidden[-1]))

@st.cache_resource
def load_models():
    # Using Auto classes for Cloud stability
    vocab = torch.load("models/vocab.pth", map_location=device) 
    lstm_model = LSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
    lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))
    lstm_model.to(device).eval()

    lr_model = joblib.load("models/logistic_regression_model.joblib")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")

    bert_tokenizer = AutoTokenizer.from_pretrained("models")
    bert_model = AutoModelForSequenceClassification.from_pretrained("models").to(device) 
    bert_model.eval()
    
    return vocab, lstm_model, lr_model, tfidf, bert_tokenizer, bert_model

with st.spinner("Loading AI Models into Cloud Memory..."):
    vocab, lstm_model, lr_model, tfidf, bert_tokenizer, bert_model = load_models()

# --- 4. PREPROCESSING & PIPELINES ---
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_pytorch_text(text, max_len=350):
    cleaned_text = text_cleaning(text)
    tokens = [vocab.get(word, vocab.get("<UNK>", 0)) for word in cleaned_text.split()]
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor([tokens]).to(device)

def lr_predict_pipeline(texts):
    features = tfidf.transform([t.lower() for t in texts])
    return lr_model.predict_proba(features)

def lstm_predict_pipeline(texts):
    probs = []
    for t in texts:
        input_tensor = preprocess_pytorch_text(t)
        with torch.no_grad():
            p = lstm_model(input_tensor).item()
            probs.append([1-p, p])
    return np.array(probs)

def bert_predict_pipeline(texts):
    inputs = bert_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

# --- 5. SESSION STATE INITIALIZATION ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analyzed_text' not in st.session_state:
    st.session_state.analyzed_text = ""

# --- 6. SIDEBAR WARNINGS ---
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

# --- 7. MAIN UI ---
st.title("🕵️‍♂️ Multi-Paradigm AI Detector")
st.markdown("Compare baseline ML, Deep Learning, and SOTA Transformers with XAI Explainability.")

user_text = st.text_area("Paste your text here:", height=200, key="current_input")

if st.button("Analyze Text", type="primary"):
    input_to_analyze = st.session_state.current_input
    
    if input_to_analyze.strip():
        with st.spinner("Analyzing across all paradigms..."):
            # Clear old XAI results from previous runs
            for m_id in ['lr', 'lstm', 'bert']:
                if f"lime_{m_id}" in st.session_state:
                    del st.session_state[f"lime_{m_id}"]
            
            # Predict Logic (Replaces API Call)
            lr_features = tfidf.transform([input_to_analyze.lower()])
            lr_probs = lr_model.predict_proba(lr_features)[0]
            
            lstm_input = preprocess_pytorch_text(input_to_analyze)
            with torch.no_grad():
                lstm_p = lstm_model(lstm_input).item()
            
            bert_inputs = bert_tokenizer(input_to_analyze, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                bert_logits = bert_model(**bert_inputs).logits
                bert_p = F.softmax(bert_logits, dim=1).cpu().numpy()[0]

            final_results = {
                "lr": {"label": "AI" if lr_probs[1] > 0.5 else "Human", "confidence": float(max(lr_probs))},
                "lstm": {"label": "AI" if lstm_p > 0.5 else "Human", "confidence": float(lstm_p if lstm_p > 0.5 else 1-lstm_p)},
                "bert": {"label": "AI" if bert_p[1] > 0.5 else "Human", "confidence": float(max(bert_p))}
            }
            
            log_to_sql(input_to_analyze, final_results)
            
            st.session_state.results = final_results
            st.session_state.analyzed_text = input_to_analyze
    else:
        st.error("Please enter some text first.")

# --- 8. DISPLAY RESULTS ---
if st.session_state.results:
    results = st.session_state.results
    original_text = st.session_state.analyzed_text
    
    st.divider()
    
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
                    # Map to the correct pipeline
                    if m_id == "lr": pipeline = lr_predict_pipeline
                    elif m_id == "bert": pipeline = bert_predict_pipeline
                    elif m_id == "lstm": pipeline = lstm_predict_pipeline
                    
                    # Generate LIME (Direct function call instead of API)
                    exp = explainer.explain_instance(original_text, pipeline, num_features=10, labels=(1,), num_samples=100)
                    st.session_state[f"lime_{m_id}"] = exp.as_html()

            # Persistent XAI Display
            if f"lime_{m_id}" in st.session_state:
                st.write("**Feature Importance:**")
                styled_html = wrap_lime_html(st.session_state[f"lime_{m_id}"])
                components.html(styled_html, height=450, scrolling=True)

    show_model_card(col1, "Logistic Regression (ML)", "lr", results['lr'])
    show_model_card(col2, "LSTM (Deep Learning)", "lstm", results['lstm'])
    show_model_card(col3, "BERT (Transformer)", "bert", results['bert'])

# --- 9. SQL LOGGING HISTORY ---
st.divider()
with st.expander("📜 View Prediction Logs (SQL Database)"):
    if st.button("Refresh History"):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 10", conn)
            conn.close()
            
            if not df.empty:
                # Format exactly like your previous layout
                df.columns = ["ID", "Text Snippet", "LR Label", "LR Conf", "LSTM Label", "LSTM Conf", "BERT Label", "BERT Conf", "Timestamp"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No queries logged yet.")
        except Exception as e:
            st.error(f"Database connection failed: {e}")