# 🖼️ Neural Image Caption Generator

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)

A deep learning model that looks at an image and describes it in a full, natural-language sentence — built as a CNN encoder + LSTM decoder with visual attention, trained from scratch, fine-tuned, evaluated, and deployed as a live web app.

**🚀 [Try it live](https://neural-image-caption-generator.streamlit.app/)** &nbsp;|&nbsp; **🤗 [Model weights on Hugging Face](https://huggingface.co/NayabGull/neural-image-caption-generator)**

---

## What this does

You upload a photo. The model generates a caption describing it — not a label, not a tag, an actual sentence — by combining a convolutional encoder (which "sees" the image) with a recurrent decoder (which "writes" about it), guided by an attention mechanism that lets the decoder focus on different parts of the image as it generates each word.

```
"a black and white dog is running through the grass"
"a group of people are sitting at a table"
```

---

## Architecture

```
Image (224×224×3)
      │
      ▼
┌─────────────────────┐
│   EfficientNetB0      │   CNN Encoder — pretrained on ImageNet,
│   (top 25 layers      │   fine-tuned end-to-end on captioning data.
│    fine-tuned)        │   Outputs a 7×7×1280 spatial feature grid
└──────────┬───────────┘   (kept spatial, not pooled, so attention
           │                has something to attend over)
           ▼
     Reshape → (49, 1280)
           │
           ▼
┌─────────────────────┐
│  Bahdanau Attention   │   At each decoding step, computes a
│  (additive attention) │   weighted context vector over the 49
└──────────┬───────────┘   spatial locations — "where to look"
           │
           ▼
┌─────────────────────┐
│   LSTM Decoder         │   Generates one word at a time, conditioned
│  (GloVe-initialized    │   on the attended image context + the
│   embeddings)          │   previously generated word
└──────────┬───────────┘
           │
           ▼
   Caption (beam search, width=3,
   with n-gram repetition blocking)
```

**Key design choices:**
- `pooling=None` on the encoder — keeps the spatial 7×7 grid instead of collapsing to one vector, which is what makes real attention possible (as opposed to a single global image vector).
- Pretrained GloVe (300d) embeddings instead of learning word vectors from scratch — gives the decoder a head start on word semantics.
- Encoder fine-tuning as a *second* training phase, at a very low learning rate (1e-5), after the decoder had already converged — lets the CNN's features adapt to the captioning task without destabilizing an untrained decoder.
- Beam search with n-gram blocking at inference — plain greedy/beam decoding produced degenerate loops like *"a man in a blue shirt and a man in a blue shirt..."*; blocking repeated n-grams fixed this.

---

## Dataset

**[Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)** — 31,783 images, ~158,000 human-written captions (5 per image), sourced via Kaggle.

Chosen for its diversity of everyday, candid scenes — a good fit for training a reasonably generalizable model at a scale that's actually trainable on free-tier compute (see *Training Infrastructure* below).

---

## Results

An earlier version of this project (v1) was trained on the smaller Flickr8k dataset with scratch-initialized embeddings and no fine-tuning. This version (v2) rebuilds the pipeline on Flickr30k with pretrained embeddings, LR scheduling, and encoder fine-tuning — validated at every stage with BLEU scores rather than assumed:

| Metric | v1 — Flickr8k baseline | v2 — Flickr30k + GloVe + fine-tuned |
|:--|:--:|:--:|
| BLEU-1 | 0.4458 | **0.5851** |
| BLEU-2 | 0.2791 | **0.3952** |
| BLEU-3 | 0.1731 | **0.2659** |
| BLEU-4 | 0.1009 | **0.1775** |

BLEU-4 (the strictest metric — exact 4-word sequence overlap) improved by roughly **80% relative** to the baseline.

**Honest limitation:** the model performs well on candid/outdoor scenes similar to Flickr30k's distribution, but is noticeably less reliable on close-up portraits, studio shots, or unusual compositions. This is an expected consequence of training on a dataset of this scale with free-tier compute — not a bug, but a real constraint worth naming rather than hiding.

---

## Training infrastructure (the part nobody tells you about)

This entire project was trained on **free-tier cloud GPUs** (Kaggle's ~30 hr/week quota, then Google Colab) — which meant training sessions disconnected mid-run more times than I'd like to admit. Rather than treat that as a blocker, I designed the pipeline to survive it:

- **Feature extraction was cached once.** Instead of re-running EfficientNetB0 over 31k images every session, extracted features were saved to disk in large batched chunks (Google Drive's mounted filesystem chokes on tens of thousands of tiny files, so features are stored as ~2,000-image `.npy` arrays with an index file mapping image → chunk).
- **Every training run was resumable.** `ModelCheckpoint` saved weights after every epoch that improved validation loss; a small bit of logic at the top of the training cell checked for an existing checkpoint and resumed from the exact last completed epoch — so a disconnect mid-training cost minutes, not hours.
- **All of this persisted to Google Drive**, not to the notebook's local/temporary disk — which is the difference between losing a day of training progress and losing nothing.

This ended up being as much a part of the project as the model architecture itself: building an ML pipeline that survives real infrastructure limitations is a different (and very real) skill from getting a model to converge on a fully available cluster.

---

## Deployment

The trained model is served through a [Streamlit](https://streamlit.io/) web app. Since the fine-tuned model's weights (~225MB) exceed GitHub's 100MB file-size limit, they're hosted separately on [Hugging Face Hub](https://huggingface.co/NayabGull/neural-image-caption-generator) and downloaded automatically the first time the app starts up.

```
neural-image-caption-generator/
├── app.py               # Streamlit UI
├── inference.py          # Model architecture (rebuilt in code) + beam search decoding
├── requirements.txt
├── runtime.txt            # pins Python version for Streamlit Cloud
├── assets/
│   ├── config.json        # vocab size, max length, embedding dims etc.
│   └── tokenizer.pkl       # fitted Keras tokenizer
├── notebook/
│   └── image-captioning-nayab-gull.ipynb   # full training pipeline
└── test_inference.py      # quick local sanity check
```

**Why the model architecture is rebuilt in code rather than loaded from a saved `.h5` model:** custom layers (a Bahdanau attention layer, an `LSTMCell`-based decoder loop) hit Keras version/serialization mismatches when saved as a full model and loaded in a different environment. Saving only the weights (`.weights.h5`) and rebuilding the exact architecture in `inference.py` sidesteps this entirely — a pattern worth knowing if you've ever hit `Unknown layer` errors loading a Keras model somewhere else.

---

## Running locally

```bash
git clone https://github.com/nayab-gull-it/neural-image-caption-generator.git
cd neural-image-caption-generator

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

The model weights will download automatically from Hugging Face on first run.

---

## Tech stack

- **Modeling:** TensorFlow / Keras 3, EfficientNetB0, GloVe embeddings
- **Training:** Kaggle & Google Colab (free-tier GPUs), Google Drive (checkpoint persistence)
- **Evaluation:** NLTK (BLEU-1 through BLEU-4)
- **Deployment:** Streamlit Community Cloud, Hugging Face Hub (weight hosting)

---

## Author

**Nayab Gull**
Final-year AI/ML/DL project
[GitHub](https://github.com/nayab-gull-it) · [Hugging Face](https://huggingface.co/NayabGull)