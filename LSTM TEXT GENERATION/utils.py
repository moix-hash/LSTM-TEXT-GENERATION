"""
utils.py - Helper functions for text cleaning, sequence creation, and prediction.
"""

import re
import numpy as np
import pickle


# ─────────────────────────────────────────────
# 1. TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase the text and strip characters that are not
    letters, digits, or common punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s',\.\!\?;:\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 2. SEQUENCE BUILDING (used during training)
# ─────────────────────────────────────────────

def build_sequences(tokenizer, corpus: str, seq_length: int = 30):
    """
    Tokenise *corpus* with an **already-fitted** Keras Tokenizer,
    then slide a window of size `seq_length + 1` over the token list.

    Returns
    -------
    X : np.ndarray  shape (n_samples, seq_length)
    y : np.ndarray  shape (n_samples,)  – raw integer labels
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # lazy import

    tokens = tokenizer.texts_to_sequences([corpus])[0]

    sequences = []
    for i in range(seq_length, len(tokens)):
        sequences.append(tokens[i - seq_length : i + 1])

    sequences = np.array(sequences)
    X = sequences[:, :-1]          # all columns except last
    y = sequences[:, -1]           # last column (target)
    return X, y


# ─────────────────────────────────────────────
# 3. TEXT GENERATION
# ─────────────────────────────────────────────

def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Draw a token index from a probability distribution scaled by *temperature*.
    temperature → 0  :  near-greedy (always pick the most likely word)
    temperature → ∞  :  uniform random
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds   = np.exp(predictions - predictions.max())   # numerically stable
    probabilities = exp_preds / exp_preds.sum()
    return np.random.choice(len(probabilities), p=probabilities)


def generate_text(
    model,
    tokenizer,
    seed_text:   str,
    next_words:  int  = 20,
    seq_length:  int  = 30,
    temperature: float = 1.0,
) -> str:
    """
    Given a seed phrase, predict *next_words* tokens one at a time.

    Parameters
    ----------
    model       : trained Keras model
    tokenizer   : fitted Keras Tokenizer (same one used during training)
    seed_text   : starting text from the user
    next_words  : how many words to generate
    seq_length  : window length the model was trained with
    temperature : creativity control (0.5 = conservative, 1.5 = creative)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # lazy import

    output_text = seed_text.strip()

    for _ in range(next_words):
        # Encode the current context
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding="pre")

        # Predict
        probs     = model.predict(token_list, verbose=0)[0]
        next_idx  = sample_with_temperature(probs, temperature)

        # Decode back to word
        next_word = tokenizer.index_word.get(next_idx, "")
        if not next_word:
            break
        output_text += " " + next_word

    return output_text


# ─────────────────────────────────────────────
# 4. SERIALISATION HELPERS
# ─────────────────────────────────────────────

def save_tokenizer(tokenizer, path: str = "models/tokenizer.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[utils] Tokenizer saved → {path}")


def load_tokenizer(path: str = "models/tokenizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
