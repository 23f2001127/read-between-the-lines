import re # Added for text cleaning
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
from fastapi.responses import HTMLResponse
import sqlite3
from datetime import datetime

# --- 1. INITIALIZATION & MODELS ---
app = FastAPI()
DB_PATH = "logs.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load Vocab and PyTorch Model
vocab = torch.load("models/vocab.pth") 
# embed_dim and hidden_dim match your Notebook 2 (Cell 13)
lstm_model = LSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
lstm_model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))
lstm_model.to(device).eval()

lr_model = joblib.load("models/logistic_regression_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

bert_tokenizer = BertTokenizer.from_pretrained("models")
bert_model = BertForSequenceClassification.from_pretrained("models").to(device) 
bert_model.eval()

explainer = LimeTextExplainer(class_names=['Human', 'AI'])

# --- 2. HELPER FUNCTIONS (MATCHING NOTEBOOK 2) ---

def text_cleaning(text):
    """Matches Notebook 2, Cell 4 exactly"""
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_pytorch_text(text, max_len=350): # Updated max_len to 350
    # 1. Clean the text
    cleaned_text = text_cleaning(text)
    
    # 2. Convert to sequence using split()
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in cleaned_text.split()]
    
    # 3. Apply POST-PADDING (Zeros at the end)
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
        
    return torch.tensor([tokens]).to(device)

# --- 3. LIME PIPELINES ---

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

# --- 4. SQL LOGGING & ROUTES ---
# (Rest of your original SQL code and routes go here...)

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

class TextRequest(BaseModel):
    text: str

@app.post("/predict_all")
async def predict_all(request: TextRequest):
    lr_features = tfidf.transform([request.text.lower()])
    lr_probs = lr_model.predict_proba(lr_features)[0]
    
    lstm_input = preprocess_pytorch_text(request.text)
    with torch.no_grad():
        lstm_p = lstm_model(lstm_input).item()
    
    bert_inputs = bert_tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        bert_logits = bert_model(**bert_inputs).logits
        bert_p = F.softmax(bert_logits, dim=1).cpu().numpy()[0]

    final_results = {
        "lr": {"label": "AI" if lr_probs[1] > 0.5 else "Human", "confidence": float(max(lr_probs))},
        "lstm": {"label": "AI" if lstm_p > 0.5 else "Human", "confidence": float(lstm_p if lstm_p > 0.5 else 1-lstm_p)},
        "bert": {"label": "AI" if bert_p[1] > 0.5 else "Human", "confidence": float(max(bert_p))}
    }
    log_to_sql(request.text, final_results)
    return final_results

@app.get("/explain")
async def explain_text(text: str, model_type: str):
    if model_type == "lr": pipeline = lr_predict_pipeline
    elif model_type == "bert": pipeline = bert_predict_pipeline
    elif model_type == "lstm": pipeline = lstm_predict_pipeline
    else: raise HTTPException(status_code=400, detail="Invalid model type")
    
    exp = explainer.explain_instance(text, pipeline, num_features=10, labels=(1,), num_samples=100)
    return HTMLResponse(content=exp.as_html())

@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return rows