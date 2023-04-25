import gradio as gr
import numpy as np
import torch 
import open_clip
import cliptag as ct

model = None
preprocess = None
device = None
text_features = None
candidate_keywords = None

FEATURES_FILE_DIR = "features/test"

def load_model():
    print("loading model")
    if torch.cuda.is_available():
        # Use the GPU (CUDA) as the device
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        # Use the CPU as the device
        device = torch.device("cpu")
        print("Using CPU")

    # Load the CLIP model and tokenizer
    print("Loading model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained='laion2b_s34b_b79k',
        device=device
    )

    return model, preprocess, device

def load_keywords(model, device):
    candidate_keywords = ct.load_keywords_from_files(FEATURES_FILE_DIR)
    text_features = ct.encode_keywords(candidate_keywords, model, device)

    return text_features, candidate_keywords

def keywords(input_image):
    labels = ct.generate_keywords_for_image(input_image, candidate_keywords, text_features, model, preprocess, device, top_k=5, batch_size=100)
    print(labels)
    return labels


model, preprocess, device = load_model()
text_features, candidate_keywords = load_keywords(model, device)

print("loading interface")
demo = gr.Interface(fn=keywords, inputs=gr.Image(type='pil', label="Image"), outputs=gr.Label(label="Image Keywords", num_top_classes=5))

print("launching interface")
demo.launch()
