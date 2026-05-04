"""
app.py  –  Streamlit frontend for the LSTM Text Generator.

Run locally:
    streamlit run app.py

On first launch the app will train the model automatically if
models/lstm_model.keras and models/tokenizer.pkl are not found.
"""

import os, sys, time, pickle
import streamlit as st
import numpy as np

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title = "LSTM Text Generator",
    page_icon  = "✍️",
    layout     = "centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&family=Lato:wght@300;400;700&display=swap');

/* ── Root palette ── */
:root {
    --ink:      #1a1a2e;
    --paper:    #f5f0e8;
    --accent:   #c9782a;
    --muted:    #7a6f5e;
    --surface:  #ede8df;
    --border:   #d4c9b5;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'Lato', sans-serif;
}

/* ── Title ── */
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-family: 'Lato', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    color: var(--muted);
    margin-bottom: 2.5rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.ornament { color: var(--accent); font-size: 1.6rem; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.8rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Output quote ── */
.output-block {
    background: var(--ink);
    color: var(--paper);
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: 1.18rem;
    line-height: 1.75;
    padding: 2rem 2.4rem;
    border-left: 4px solid var(--accent);
    border-radius: 2px;
    margin-top: 1rem;
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* ── Seed word highlight ── */
.seed-word { color: #f0b662; font-weight: bold; font-style: normal; }

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.92rem !important;
    color: var(--ink) !important;
}
.stSlider > div { accent-color: var(--accent) !important; }
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 2rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox > div { background: #fff !important; border-radius: 3px !important; }

/* ── Status / warning ── */
.stAlert { border-radius: 3px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-family: 'Lato', sans-serif;
    letter-spacing: 0.05em;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "lstm_model.keras")
TOK_PATH    = os.path.join("models", "tokenizer.pkl")
SEQ_LENGTH  = 30   # must match what was used during training

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the Keras model and tokenizer from disk."""
    from tensorflow.keras.models import load_model
    model     = load_model(MODEL_PATH)
    with open(TOK_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def models_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(TOK_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION HELPER (thin wrapper around utils.generate_text)
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(model, tokenizer, seed: str, n_words: int, temp: float) -> str:
    from utils import generate_text
    return generate_text(
        model       = model,
        tokenizer   = tokenizer,
        seed_text   = seed,
        next_words  = n_words,
        seq_length  = SEQ_LENGTH,
        temperature = temp,
    )


def highlight_seed(full_text: str, seed: str) -> str:
    """Wrap seed words in a styled span inside the output."""
    if full_text.lower().startswith(seed.lower()):
        rest = full_text[len(seed):]
        return f'<span class="seed-word">{seed}</span>{rest}'
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">✍️ LSTM Text Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Shakespeare • Long Short-Term Memory • Neural Language Model</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ON FIRST RUN IF NEEDED
# ─────────────────────────────────────────────────────────────────────────────
if not models_exist():
    st.warning("⚠️  No trained model found. Running training pipeline — this may take a few minutes on CPU.")
    with st.spinner("🔧 Training LSTM on Shakespeare corpus …"):
        os.makedirs("models", exist_ok=True)
        os.makedirs("data",   exist_ok=True)
        # Import and run training inline
        import importlib, train as train_module
        importlib.reload(train_module)   # ensures a clean run
        raw           = train_module.fetch_data()
        X, y_cat, tok, vocab_size = train_module.preprocess(raw)
        mdl           = train_module.build_model(vocab_size, SEQ_LENGTH)
        train_module.train(mdl, X, y_cat)
        from utils import save_tokenizer
        save_tokenizer(tok, TOK_PATH)
    st.success("✅ Training complete! Model and tokenizer saved.")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model …"):
    model, tokenizer = load_artifacts()

vocab_size = len(tokenizer.word_index) + 1

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.metric("Vocabulary size", f"{vocab_size:,}")
    st.metric("Sequence length", SEQ_LENGTH)
    total_params = model.count_params()
    st.metric("Total parameters", f"{total_params:,}")

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown(
        "1. Your seed text is tokenised into integers.  \n"
        "2. The LSTM processes the sequence and predicts a probability distribution over the vocabulary.  \n"
        "3. A word is sampled using the **temperature** setting.  \n"
        "4. The new word is appended and the loop repeats."
    )
    st.markdown("---")
    st.markdown("### 🌡️ Temperature Guide")
    st.markdown(
        "| Setting | Effect |\n"
        "|---------|--------|\n"
        "| 0.5     | Conservative, predictable |\n"
        "| 1.0     | Balanced |\n"
        "| 1.5     | Creative, unpredictable |"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚙️ Generation Settings</div>', unsafe_allow_html=True)

seed_text = st.text_input(
    "Seed text",
    value    = "to be or not to be",
    help     = "The starting phrase — the model will continue from here.",
)

col1, col2 = st.columns(2)
with col1:
    next_words = st.slider("Words to generate", min_value=5, max_value=100, value=30, step=5)
with col1:
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                            help="Lower = safer, Higher = more creative")
with col2:
    n_variations = st.selectbox("Variations", options=[1, 2, 3], index=0,
                                help="Generate multiple outputs at once for comparison")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────────────────────────────────────
if st.button("✦ Generate Text", use_container_width=True):
    if not seed_text.strip():
        st.warning("Please enter a seed phrase to begin.")
    else:
        with st.spinner("Composing …"):
            results = []
            for i in range(n_variations):
                generated = run_generation(model, tokenizer, seed_text.strip(), next_words, temperature)
                results.append(generated)
                time.sleep(0.05)   # tiny pause for UX feel

        st.markdown("### 📜 Generated Text")
        for idx, text in enumerate(results):
            label = f"Variation {idx + 1}" if n_variations > 1 else ""
            if label:
                st.caption(label)
            html = highlight_seed(text, seed_text.strip())
            st.markdown(f'<div class="output-block">{html}</div>', unsafe_allow_html=True)
            st.code(text, language=None)   # plain-text copy box

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SEEDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**💡 Try these seeds:**")
example_seeds = [
    "all the world's a stage",
    "shall i compare thee",
    "to be or not to be",
    "what a piece of work is a man",
    "the lady doth protest",
]
cols = st.columns(len(example_seeds))
for col, seed in zip(cols, example_seeds):
    with col:
        if st.button(f'"{seed[:20]}…"', key=seed, use_container_width=True):
            st.session_state["_seed_copy"] = seed
            st.info(f"Copied seed: *{seed}* — paste it into the Seed text box above.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Built with TensorFlow · Keras · Streamlit &nbsp;|&nbsp; '
    'Trained on the Complete Works of Shakespeare (Project Gutenberg)'
    '</div>',
    unsafe_allow_html=True,
)
