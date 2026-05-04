#  LSTM Text Generator

A Streamlit web-app that uses a two-layer LSTM trained on Shakespeare's complete works to continue any seed phrase you provide.

---

## Project Structure

```
text-generator-lstm/
├── data/
│   └── raw_text.txt           # Downloaded automatically on first run
├── models/
│   ├── lstm_model.keras       # Saved trained model
│   └── tokenizer.pkl          # Saved Keras Tokenizer
├── notebook/
│   └── training_script.ipynb  # Step-by-step exploration & training
├── app.py                     # Streamlit application
├── train.py                   # Standalone training script
├── utils.py                   # Shared helpers (clean, build_sequences, generate)
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Clone / unzip the project
cd text-generator-lstm

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-train the model — the app does this automatically if skipped
python train.py

# 5. Launch the app
streamlit run app.py
```

The first launch will **automatically download the Shakespeare corpus and train the model** if `models/` is empty. Training takes ~10–20 min on CPU (much faster on GPU).

---

# Deploy to Streamlit Cloud
---

## Key Hyperparameters (edit `train.py`)

| Parameter     | Default | Description |
|---------------|---------|-------------|
| `SEQ_LENGTH`  | 30      | Context window — how many previous tokens the model sees |
| `EMBED_DIM`   | 100     | Embedding dimension |
| `LSTM1_UNITS` | 150     | Units in first LSTM layer |
| `LSTM2_UNITS` | 100     | Units in second LSTM layer |
| `DROPOUT`     | 0.2     | Dropout rate (regularisation) |
| `BATCH_SIZE`  | 128     | Training batch size |
| `EPOCHS`      | 50      | Max epochs (EarlyStopping will cut short) |

---

## Architecture Overview

```
Input  →  Embedding(vocab, 100)
       →  LSTM(150, return_sequences=True)  →  Dropout(0.2)
       →  LSTM(100)                          →  Dropout(0.2)
       →  Dense(vocab, softmax)
```

Loss: `categorical_crossentropy` | Optimiser: `Adam`

---

## Tips

- **Dataset**: The Shakespeare corpus works well because of its repetitive structure. For a different style, drop any `.txt` file into `data/raw_text.txt` and retrain.
- **Overfitting**: Watch the training loss curve in the notebook. Stop when it plateaus.
- **Temperature**: Values around `0.8–1.2` give the best balance between coherence and creativity.
- **Git LFS**: Model files can exceed 100 MB. Install Git LFS before committing:  
  ```bash
  git lfs install
  git lfs track "models/*.keras" "models/*.pkl"
  git add .gitattributes
  ```
