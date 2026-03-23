import streamlit as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# --- 1. CONFIG & UI ---
st.set_page_config(page_title="AI vs Human Detector Pro", layout="wide")
st.title("Multi-Paradigm AI Detector")
st.markdown("Compare baseline ML, Deep Learning, and SOTA Transformers.")

# --- 2. MODEL DEFINITIONS & CACHING ---
# We use CPU for the free cloud tier
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
    """Loads all models once and caches them to save Cloud RAM."""
    model_path = "models"
    
    # 1. Load LR
    lr = joblib.load(f"{model_path}/logistic_regression_model.joblib")
    tfidf = joblib.load(f"{model_path}/tfidf_vectorizer.joblib")
    
    # 2. Load LSTM
    vocab = torch.load(f"{model_path}/vocab.pth", map_location=device)
    lstm = LSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
    lstm.load_state_dict(torch.load(f"{model_path}/lstm_model.pth", map_location=device))
    lstm.to(device).eval()
    
    # 3. Load BERT (Using Auto-classes for better configuration matching)
    bert_tok = AutoTokenizer.from_pretrained(model_path)
    bert_mod = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    bert_mod.eval()
    
    return lr, tfidf, lstm, vocab, bert_tok, bert_mod

with st.spinner("Loading AI Models into Cloud Memory... This takes a moment on first boot."):
    lr_model, tfidf, lstm_model, vocab, bert_tokenizer, bert_model = load_models()

# --- 3. HELPER FUNCTIONS ---
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

# --- 4. MAIN APP LOGIC ---
user_text = st.text_area("Paste your text here:", height=200)

if st.button("Analyze Text", type="primary"):
    if user_text.strip():
        with st.spinner("Analyzing text across all paradigms..."):
            
            # LR Prediction
            lr_features = tfidf.transform([user_text.lower()])
            lr_probs = lr_model.predict_proba(lr_features)[0]
            lr_conf = float(max(lr_probs))
            lr_label = "AI" if lr_probs[1] > 0.5 else "Human"
            
            # LSTM Prediction
            lstm_input = preprocess_pytorch_text(user_text)
            with torch.no_grad():
                lstm_p = lstm_model(lstm_input).item()
            lstm_conf = float(lstm_p if lstm_p > 0.5 else 1-lstm_p)
            lstm_label = "AI" if lstm_p > 0.5 else "Human"
            
            # BERT Prediction
            bert_inputs = bert_tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                bert_logits = bert_model(**bert_inputs).logits
                bert_p = F.softmax(bert_logits, dim=1).numpy()[0]
            bert_conf = float(max(bert_p))
            bert_label = "AI" if bert_p[1] > 0.5 else "Human"

            # Display Results
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