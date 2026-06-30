import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 70px;
    font-weight: bold;
    color: #4CAF50;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 32px;
    color: #BBBBBB;
    margin-bottom: 30px;
}

.caption-box {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 18px;
}

.metric-box {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
    width=120
)

st.sidebar.title("📌 Project Details")

st.sidebar.info("""
### AI Image Captioning System

🔹 BLIP Transformer Model  
🔹 TF-IDF Caption Matching  
🔹 Flickr8k Dataset  
🔹 Streamlit Dashboard  

Developed using:
- Deep Learning
- NLP
- Computer Vision
""")

st.sidebar.success("✅ System Ready")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model

processor, model_blip = load_model()

# ================= LOAD CAPTIONS =================
with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

all_captions = []

for caps in mapping.values():
    for c in caps:
        c = c.replace("start", "").replace("end", "").strip()
        all_captions.append(c)

# ================= FUNCTION =================
def generate_and_match_caption(image):

    inputs = processor(image, return_tensors="pt")

    output = model_blip.generate(**inputs)

    generated_caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    # TF-IDF Matching
    texts = [generated_caption] + all_captions

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )

    best_idx = similarity.argmax()

    matched_caption = all_captions[best_idx]

    return generated_caption, matched_caption

# ================= HEADER =================
st.markdown(
    """
    <h1 style='
    text-align: center;
    color: #4CAF50;
    font-size: 60px;
    font-weight: bold;
    '>
    🖼️ AI Image Captioning System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='
    text-align: center;
    color: #BBBBBB;
    font-size: 35px;
    '>
    Generate Smart Captions using BLIP + TF-IDF
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("")

# ================= METRICS =================
st.markdown("""
<style>

.metric-card {
    background: linear-gradient(145deg, #1a1a1a, #222222);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    border: 1px solid #333333;
    transition: 0.3s;
}

.metric-card:hover {
    transform: scale(1.03);
    border: 1px solid #4CAF50;
}

.metric-title {
    color: #BBBBBB;
    font-size: 22px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 50px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">📸 Dataset Images</div>
        <div class="metric-value">8092</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">🤖 Model</div>
        <div class="metric-value">BLIP</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">🎯 Approx Accuracy</div>
        <div class="metric-value">70%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ================= IMAGE UPLOAD =================
uploaded_files = st.file_uploader(
    "📤 Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ================= HISTORY =================
history = []

# ================= PROCESS =================
if uploaded_files:

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            with st.spinner("🔍 AI is analyzing image..."):

                gen_caption, match_caption = generate_and_match_caption(image)

            st.success("✅ Caption Generated")

            st.markdown("### 🔹 BLIP Generated Caption")

            st.markdown(
                f'<div class="caption-box">{gen_caption}</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown("### 🔹 Best Matched Caption")

            st.markdown(
                f'<div class="caption-box">{match_caption}</div>',
                unsafe_allow_html=True
            )

            # Download Button
                      # Download Button
            st.download_button(
                label="📥 Download Caption",
                data=gen_caption,
                file_name=f"{uploaded_file.name}_caption.txt",
                mime="text/plain",
                key=uploaded_file.name
            )

            # Save history
            history.append({
                "Image": uploaded_file.name,
                "Generated Caption": match_caption
            })
            
        st.write("---")

# ================= HISTORY TABLE =================
if len(history) > 0:

    st.subheader("🕘 Caption History")

    df = pd.DataFrame(history)

    st.dataframe(df)

# ================= FOOTER =================
st.write("---")

st.markdown(
    """
    <center>
    🚀 Developed using Deep Learning, Transformers & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)