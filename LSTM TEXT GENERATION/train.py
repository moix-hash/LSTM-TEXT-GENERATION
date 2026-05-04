"""
train.py  –  Data preprocessing, LSTM training, and model/tokenizer export.

Run from the project root:
    python train.py

After completion you will find:
    models/lstm_model.keras
    models/tokenizer.pkl
"""

import os, pickle, urllib.request
import numpy as np

# ── TF / Keras ──────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.layers    import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models    import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from utils import clean_text, build_sequences, save_tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# HYPER-PARAMETERS  (edit freely)
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LENGTH   = 30          # look-back window (tokens)
EMBED_DIM    = 100         # embedding size
LSTM1_UNITS  = 150
LSTM2_UNITS  = 100
DROPOUT      = 0.2
BATCH_SIZE   = 128
EPOCHS       = 50          # EarlyStopping will cut this short when needed

DATA_URL  = "https://www.gutenberg.org/files/100/100-0.txt"   # Shakespeare (Project Gutenberg)
DATA_PATH = "data/raw_text.txt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
TOK_PATH   = os.path.join(MODEL_DIR, "tokenizer.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 1 · FETCH / LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data() -> str:
    os.makedirs("data",    exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"[train] Downloading dataset → {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("[train] Download complete.")
    else:
        print(f"[train] Using cached data at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Trim Project Gutenberg boilerplate (optional but recommended)
    start = raw.find("THE SONNETS")
    if start == -1:
        start = 0
    raw = raw[start : start + 300_000]   # use first ~300 k chars to keep training fast
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 2 · PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(raw: str):
    print("[train] Cleaning text …")
    corpus = clean_text(raw)

    print("[train] Fitting tokenizer …")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[train] Vocabulary size: {vocab_size:,} unique tokens")

    print("[train] Building sequences …")
    X, y = build_sequences(tokenizer, corpus, SEQ_LENGTH)
    print(f"[train] Training samples: {len(X):,}")

    y_cat = to_categorical(y, num_classes=vocab_size)
    return X, y_cat, tokenizer, vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# 3 · BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, seq_length: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=seq_length),
        LSTM(LSTM1_UNITS, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(LSTM2_UNITS),
        Dropout(DROPOUT),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        loss      = "categorical_crossentropy",
        optimizer = "adam",
        metrics   = ["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4 · TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(model, X, y_cat):
    callbacks = [
        EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="loss"),
    ]
    history = model.fit(
        X, y_cat,
        epochs     = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks  = callbacks,
        verbose    = 1,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw_text               = fetch_data()
    X, y_cat, tokenizer, vocab_size = preprocess(raw_text)
    model                  = build_model(vocab_size, SEQ_LENGTH)
    history                = train(model, X, y_cat)

    # Save tokenizer (model is saved by ModelCheckpoint)
    save_tokenizer(tokenizer, TOK_PATH)

    final_loss = history.history["loss"][-1]
    final_acc  = history.history["accuracy"][-1]
    print(f"\n[train] ✓ Training complete  |  loss={final_loss:.4f}  acc={final_acc:.4f}")
    print(f"[train] Model  → {MODEL_PATH}")
    print(f"[train] Tokenizer → {TOK_PATH}")
