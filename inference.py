"""
inference.py
-------------
Core inference logic for the Neural Image Caption Generator (v2).

Architecture: EfficientNetB0 encoder (fine-tuned, top 25 layers unfrozen)
+ Bahdanau attention + LSTM decoder with GloVe-initialized embeddings.
"""

import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTMCell, Dropout, Concatenate, Layer, Reshape
)
from tensorflow.keras.models import Model

ASSETS_DIR = Path(__file__).parent / "assets"

# Fine-tuned weights are too large for GitHub (225MB > 100MB limit),
# so they're hosted on Hugging Face Hub and downloaded at runtime instead.
WEIGHTS_URL = "https://huggingface.co/NayabGull/neural-image-caption-generator/resolve/main/best_weights_finetuned.weights.h5"
WEIGHTS_PATH = ASSETS_DIR / "best_weights_finetuned.weights.h5"


def ensure_weights_downloaded():
    """Downloads model weights from Hugging Face if not already present locally."""
    if not WEIGHTS_PATH.exists():
        ASSETS_DIR.mkdir(exist_ok=True)
        print("Downloading model weights from Hugging Face...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("Download complete.")


class BahdanauAttention(Layer):
    """Additive attention over the encoder's 49 spatial feature locations."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        return super().get_config()


class CaptionDecoder(Layer):
    """
    Runs the per-timestep attention + LSTMCell loop. Wrapped in a Layer
    because Keras 3 disallows raw TF ops directly on KerasTensors outside
    a Layer's call().
    """

    def __init__(self, lstm_units, attention_units, vocab_size, timesteps, **kwargs):
        super().__init__(**kwargs)
        self.lstm_units = lstm_units
        self.timesteps = timesteps
        self.attention = BahdanauAttention(attention_units)
        self.lstm_cell = LSTMCell(lstm_units)
        self.concat = Concatenate(axis=-1)
        self.output_dense1 = Dense(lstm_units, activation="relu")
        self.dropout = Dropout(0.5)
        self.output_dense2 = Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        encoder_output, decoder_embedding = inputs
        batch_size = tf.shape(encoder_output)[0]
        hidden_state = tf.zeros((batch_size, self.lstm_units))
        cell_state = tf.zeros((batch_size, self.lstm_units))
        states = [hidden_state, cell_state]

        all_outputs = []
        for t in range(self.timesteps):
            context_vector, _ = self.attention(encoder_output, hidden_state)
            word_embedding_t = decoder_embedding[:, t, :]
            lstm_input = self.concat([context_vector, word_embedding_t])
            lstm_out, states = self.lstm_cell(lstm_input, states, training=training)
            hidden_state = states[0]
            out = self.output_dense1(lstm_out)
            out = self.dropout(out, training=training)
            out = self.output_dense2(out)
            all_outputs.append(tf.expand_dims(out, axis=1))

        return tf.concat(all_outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"lstm_units": self.lstm_units, "timesteps": self.timesteps})
        return config


def build_finetuned_model(vocab_size: int, max_length: int,
                           embedding_dim: int = 300, lstm_units: int = 512,
                           attention_units: int = 512, feature_dim: int = 1280) -> Model:
    """
    Rebuilds the exact fine-tuned end-to-end graph used in training:
    raw image -> EfficientNetB0 (unfrozen top layers) -> spatial features
    -> Bahdanau attention + LSTM decoder -> caption.
    """
    decoder_timesteps = max_length - 1

    cnn_base = EfficientNetB0(weights=None, include_top=False, pooling=None,
                               input_shape=(224, 224, 3))

    raw_image_input = Input(shape=(224, 224, 3), name="raw_image")
    cnn_features = cnn_base(raw_image_input)
    cnn_features_reshaped = Reshape((49, feature_dim))(cnn_features)

    encoder_dense = Dense(lstm_units, activation="relu", name="encoder_projection")
    encoder_output = encoder_dense(cnn_features_reshaped)

    decoder_input = Input(shape=(decoder_timesteps,), name="caption_input")
    embedding_layer = Embedding(
        input_dim=vocab_size, output_dim=embedding_dim,
        mask_zero=True, name="glove_embedding"
    )
    decoder_embedding = embedding_layer(decoder_input)

    decoder = CaptionDecoder(lstm_units, attention_units, vocab_size, decoder_timesteps,
                              name="caption_decoder")
    final_output = decoder([encoder_output, decoder_embedding])

    return Model(inputs=[raw_image_input, decoder_input], outputs=final_output,
                 name="caption_generator_v2_finetune")


class CaptionGenerator:
    """
    Loads the trained fine-tuned model once and generates captions for any
    PIL image passed to it.
    """

    def __init__(self, assets_dir: Path = ASSETS_DIR):
        ensure_weights_downloaded()

        with open(assets_dir / "config.json", "r") as f:
            self.config = json.load(f)

        with open(assets_dir / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        self.max_length = self.config["max_length"]

        self.model = build_finetuned_model(
            vocab_size=self.config["vocab_size"],
            max_length=self.max_length,
            embedding_dim=self.config.get("embedding_dim", 300),
            lstm_units=self.config.get("lstm_units", 512),
            attention_units=self.config.get("attention_units", 512),
            feature_dim=self.config.get("feature_dim", 1280),
        )
        self.model.load_weights(WEIGHTS_PATH)

        self.index_to_word = {idx: w for w, idx in self.tokenizer.word_index.items()}
        self.start_idx = self.tokenizer.word_index["startseq"]
        self.end_idx = self.tokenizer.word_index["endseq"]

    def _preprocess_image(self, pil_image) -> np.ndarray:
        img = pil_image.convert("RGB").resize((224, 224))
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def generate_caption(self, pil_image, beam_width: int = 3,
                          no_repeat_ngram_size: int = 3) -> str:
        image_arr = self._preprocess_image(pil_image)
        beams = [([self.start_idx], 0.0)]

        for _ in range(self.max_length - 1):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == self.end_idx:
                    all_candidates.append((seq, score))
                    continue

                padded_seq = pad_sequences([seq], maxlen=self.max_length - 1, padding="post")
                preds = self.model.predict([image_arr, padded_seq], verbose=0)
                next_pos = min(len(seq) - 1, preds.shape[1] - 1)
                probs = preds[0, next_pos, :]

                top_indices = np.argsort(probs)[-beam_width:]
                for idx in top_indices:
                    if idx == 0:
                        continue
                    candidate_seq = seq + [int(idx)]

                    if len(candidate_seq) >= no_repeat_ngram_size:
                        last_ngram = tuple(candidate_seq[-no_repeat_ngram_size:])
                        earlier = [tuple(candidate_seq[i:i + no_repeat_ngram_size])
                                   for i in range(len(candidate_seq) - no_repeat_ngram_size)]
                        if last_ngram in earlier:
                            continue

                    candidate_score = score + float(np.log(probs[idx] + 1e-10))
                    all_candidates.append((candidate_seq, candidate_score))

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == self.end_idx for seq, _ in beams):
                break

        best_seq = beams[0][0]
        words = [self.index_to_word.get(idx, "") for idx in best_seq]
        words = [w for w in words if w not in ("startseq", "endseq", "")]
        caption = " ".join(words).strip()
        return caption.capitalize()


_generator_instance = None


def get_generator() -> CaptionGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = CaptionGenerator()
    return _generator_instance