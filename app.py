"""
app.py
-------
Streamlit UI for the Neural Image Caption Generator.
"""

import streamlit as st
from PIL import Image
from inference import get_generator

st.set_page_config(
    page_title="Neural Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    h1, h2, h3 {
        color: #e6edf3;
        font-weight: 700;
    }

    .title-accent {
        color: #2dd4bf;
    }

    .caption-box {
        background-color: #161b22;
        border: 1px solid #2dd4bf;
        border-radius: 8px;
        padding: 24px;
        margin-top: 16px;
        font-size: 20px;
        color: #f59e0b;
        text-align: center;
        line-height: 1.5;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #161b22;
        border: 2px dashed #30363d;
        border-radius: 8px;
    }

    /* FIX: file uploader instruction text + "Drag and drop" text visibility */
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: #c9d1d9 !important;
    }

    /* FIX: general markdown/paragraph text visibility */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #c9d1d9;
    }

    /* FIX: slider label text */
    .stSlider label p {
        color: #c9d1d9 !important;
    }

    /* FIX: "Browse files" button text */
    [data-testid="stFileUploaderDropzone"] button {
        color: #0d1117 !important;
    }

    .stButton > button {
        background-color: #2dd4bf;
        color: #0d1117;
        font-weight: 700;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
    }
    .stButton > button:hover {
        background-color: #f59e0b;
        color: #0d1117;
    }

    div[data-testid="stMetricValue"] {
        color: #2dd4bf;
    }
    div[data-testid="stMetricLabel"] {
        color: #c9d1d9;
    }

    hr {
        border-color: #30363d;
    }

    .footer {
        text-align: center;
        color: #8b949e;
        font-size: 13px;
        margin-top: 48px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    return get_generator()


with st.sidebar:
    st.markdown("## 🖼️ About This Project")
    st.markdown(
        "A **CNN + LSTM encoder-decoder** model with **Bahdanau attention** "
        "that generates natural-language captions for any image."
    )

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        "- **Encoder:** EfficientNetB0 (fine-tuned, top 25 layers)\n"
        "- **Decoder:** LSTM + Bahdanau Attention\n"
        "- **Embeddings:** Pretrained GloVe (300d)\n"
        "- **Decoding:** Beam search + n-gram repetition blocking"
    )

    st.markdown("---")
    st.markdown("### Training Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset", "Flickr30k")
        st.metric("Images", "31,783")
    with col2:
        st.metric("Vocab Size", "19,097")
        st.metric("Params", "23.7M")

    st.markdown("---")
    st.markdown("### Evaluation (BLEU)")
    st.markdown(
        "| Metric | Score |\n"
        "|---|---|\n"
        "| BLEU-1 | 0.585 |\n"
        "| BLEU-2 | 0.395 |\n"
        "| BLEU-3 | 0.266 |\n"
        "| BLEU-4 | 0.178 |"
    )

    st.markdown("---")
    st.markdown(
        "<div class='footer'>Built with TensorFlow + Streamlit<br>"
        "<a href='https://github.com/YOUR_USERNAME/neural-image-caption-generator' "
        "style='color:#2dd4bf;'>View on GitHub</a></div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<h1>Neural Image <span class='title-accent'>Caption</span> Generator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "Upload an image and the model will generate a natural-language description "
    "using a CNN-LSTM encoder-decoder with visual attention."
)
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    beam_width = st.slider(
        "Beam width (higher = more thorough search, slower)",
        min_value=1, max_value=5, value=3
    )

    generate_clicked = False
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        generate_clicked = st.button("✨ Generate Caption", use_container_width=True)

with col_right:
    st.markdown("### 📝 Generated Caption")

    if uploaded_file is None:
        st.markdown(
            "<div class='caption-box' style='color:#8b949e;'>"
            "Upload an image to get started.</div>",
            unsafe_allow_html=True
        )
    elif generate_clicked:
        with st.spinner("Analyzing image and generating caption..."):
            try:
                generator = load_model()
                caption = generator.generate_caption(image, beam_width=beam_width)
                st.markdown(
                    f"<div class='caption-box'>\"{caption}\"</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Something went wrong while generating the caption: {e}")
    else:
        st.markdown(
            "<div class='caption-box' style='color:#8b949e;'>"
            "Click 'Generate Caption' to see the result.</div>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown(
    "<div class='footer'>Note: This model was trained on Flickr30k (candid/outdoor scenes). "
    "It performs best on similar image types and may be less accurate on close-up portraits, "
    "studio shots, or unusual compositions.</div>",
    unsafe_allow_html=True
)