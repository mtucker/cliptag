import gradio as gr
import numpy as np
import torch 
import open_clip
from cliptag import ImageKeywordGenerator

model = None
preprocess = None
device = None

FEATURES_FILE_DIR = "features/test"
TOP_K_LABELS = 10

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

ikg = ImageKeywordGenerator(model, preprocess, device, FEATURES_FILE_DIR)

def keywords(input_image):
    labels = ikg.generate_keywords_for_image(input_image, top_k=TOP_K_LABELS)
    return labels


print("loading interface")
demo = gr.Interface(fn=keywords, inputs=gr.Image(type='pil', label="Image"), outputs=gr.Label(label="Image Keywords", num_top_classes=TOP_K_LABELS))

if __name__ == "__main__":
    print("launching interface")
    demo.launch()
