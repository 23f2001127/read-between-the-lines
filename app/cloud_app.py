import streamlit as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import sqlite3
from datetime import datetime
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="AI vs Human Detector Pro", layout="wide", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Multi-Paradigm AI Detector (Cloud Version)")
st.markdown("Compare baseline ML, Deep Learning, and SOTA Transformers.")

# --- 2. SIDEBAR NOTES ---
with st.sidebar:
    st.header("📝 Project Notes")
    st.info(
        "**Architecture Audit**\n\n"
        "1. **Baseline (LR):** High speed, low carbon footprint.\n"
        "2. **Deep Learning (LSTM):** Captures sequential memory, uses custom post-padding.\n"
        "3. **SOTA (BERT):** 110M parameters. Heavily penalized by Generalization Gaps on modern LLM text."
    )
    st.divider()
    st.markdown("*(Developed by Antareep Ghosh)*")

# --- 3. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries 
                 (timestamp TEXT, text_snippet TEXT, lr_pred TEXT, lstm_pred TEXT, bert_pred TEXT)''')
    conn.commit()
    conn.close()

init_db()

def log_to_db(text, lr, lstm, bert):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    snippet = text[:50] + "..." if len(text) > 50 else text
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?)", (timestamp, snippet, lr, lstm, bert))
    conn.commit()
    conn.close()

# --- 4. MODEL DEFINITIONS & CACHING ---
device = torch.device("cpu")

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
    model_path = "models"
    lr = joblib.load(f"{model_path}/logistic_regression_model.joblib")
    tfidf = joblib.load(f"{model_path}/tfidf_vectorizer.joblib")
    
    vocab = torch.load(f"{model_path}/vocab.pth", map_location=device)
    lstm = LSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
    lstm.load_state_dict(torch.load(f"{model_path}/lstm_model.pth", map_location=device))
    lstm.to(device).eval()
    
    bert_tok = AutoTokenizer.from_pretrained(model_path)
    bert_mod = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    bert_mod.eval()
    
    return lr, tfidf, lstm, vocab, bert_tok, bert_mod

with st.spinner("Loading AI Models into Cloud Memory..."):
    lr_model, tfidf, lstm_model, vocab, bert_tokenizer, bert_model = load_models()

# --- 5. HELPER FUNCTIONS ---
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

def lr_predict_proba_wrapper(texts):
    """Wrapper function for LIME to interact with the LR model"""
    features = tfidf.transform(texts)
    return lr_model.predict_proba(features)

# --- 6. MAIN APP LOGIC (TABS) ---
tab1, tab2, tab3 = st.tabs(["🕵️‍♂️ Detection", "🧠 Explainability (XAI)", "📜 History Log"])

with tab1:
    user_text = st.text_area("Paste text to analyze:", height=200, key="main_text")
    
    if st.button("Analyze Text", type="primary"):
        if user_text.strip():
            with st.spinner("Analyzing text across all paradigms..."):
                
                # LR
                lr_features = tfidf.transform([user_text.lower()])
                lr_probs = lr_model.predict_proba(lr_features)[0]
                lr_conf = float(max(lr_probs))
                lr_label = "AI" if lr_probs[1] > 0.5 else "Human"
                
                # LSTM
                lstm_input = preprocess_pytorch_text(user_text)
                with torch.no_grad():
                    lstm_p = lstm_model(lstm_input).item()
                lstm_conf = float(lstm_p if lstm_p > 0.5 else 1-lstm_p)
                lstm_label = "AI" if lstm_p > 0.5 else "Human"
                
                # BERT
                bert_inputs = bert_tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    bert_logits = bert_model(**bert_inputs).logits
                    bert_p = F.softmax(bert_logits, dim=1).numpy()[0]
                bert_conf = float(max(bert_p))
                bert_label = "AI" if bert_p[1] > 0.5 else "Human"

                # Log to SQLite
                log_to_db(user_text, f"{lr_label} ({lr_conf:.0%})", f"{lstm_label} ({lstm_conf:.0%})", f"{bert_label} ({bert_conf:.0%})")

                # UI Display
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                def render_metric(col, title, label, conf):
                    with col:
                        st.subheader(title)
                        color = "red" if label == "AI" else "green"
                        st.markdown(f"### Status: :{color}[{label}]")
                        st.metric("Confidence", f"{conf:.2%}")
                        st.progress(conf)

                render_metric(col1, "Logistic Regression", lr_label, lr_conf)
                render_metric(col2, "LSTM", lstm_label, lstm_conf)
                render_metric(col3, "BERT", bert_label, bert_conf)
                
        else:
            st.error("Please enter some text first.")

with tab2:
    st.subheader("LIME (Local Interpretable Model-Agnostic Explanations)")
    st.markdown("*(Note: Explainability is currently running on the Baseline LR model to ensure fast cloud latency)*")
    
    xai_text = st.text_area("Paste text to explain why it was flagged:", height=150, key="xai_text")
    if st.button("Generate Explanation"):
        if xai_text.strip():
            with st.spinner("Generating LIME Explanation (This may take a few seconds)..."):
                explainer = LimeTextExplainer(class_names=["Human", "AI"])
                exp = explainer.explain_instance(xai_text, lr_predict_proba_wrapper, num_features=10)
                
                # Render the HTML output from LIME
                components.html(exp.as_html(), height=400, scrolling=True)
        else:
            st.warning("Enter text to explain.")

with tab3:
    st.subheader("Recent Queries")
    st.caption("Logs are stored in an ephemeral SQLite database. They will reset when the cloud server sleeps.")
    
    conn = sqlite3.connect("history.db")
    import pandas as pd
    try:
        df = pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp DESC LIMIT 20", conn)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info("No queries logged yet.")
    finally:
        conn.close()