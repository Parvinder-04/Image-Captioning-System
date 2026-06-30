# Image Captioning System

## Project Overview

This project is a Deep Learning based Image Captioning System that automatically generates captions for images using the BLIP (Bootstrapping Language Image Pretraining) Transformer model. The generated caption is then compared with Flickr8k dataset captions using TF-IDF and Cosine Similarity to find the most relevant caption.


## Features

- Upload any image
- AI-generated caption using BLIP
- Best matching caption from Flickr8k dataset
- Modern Streamlit UI
- Download generated caption
- Caption history


## 🛠 Technologies Used

- Python
- TensorFlow
- PyTorch
- BLIP Transformer
- Hugging Face Transformers
- Streamlit
- Scikit-learn
- PIL
- Matplotlib


## Dataset

- Flickr8k Dataset
- 8,000+ images
- 5 captions per image


## Workflow

1. Upload Image
2. BLIP generates caption
3. TF-IDF compares with dataset captions
4. Cosine Similarity finds best match
5. Display final caption


## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the application

```bash
streamlit run app.py
```


## Developed By

Parvinder Singh Dhull

Niral Saxena
