import gradio as gr
import numpy as np
import torch 
import open_clip
from cliptag import ImageKeywordGenerator
from cliptag.cliptag import CIImageKeywordGenerator
from clip_interrogator import Config

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

# # Load the CLIP model and tokenizer
# print("Loading model...")

# model, _, preprocess = open_clip.create_model_and_transforms(
#     'ViT-B-32', 
#     pretrained='laion2b_s34b_b79k',
#     device=device
# )

# ikg = ImageKeywordGenerator(model, preprocess, device, FEATURES_FILE_DIR)
cikg = CIImageKeywordGenerator(FEATURES_FILE_DIR, Config(caption_model_name=None, device=device))

config2 = Config(clip_model_name="ViT-B-32/laion2b_s34b_b79k", caption_model_name=None, device=device)
cikg2 = CIImageKeywordGenerator(FEATURES_FILE_DIR, config2)

def keywords(input_image):
    # ikg_labels = ikg.generate_keywords_for_image(input_image, top_k=TOP_K_LABELS)
    cikg_labels = cikg.generate_keywords_for_image(input_image, top_k=TOP_K_LABELS)
    cikg2_labels = cikg2.generate_keywords_for_image(input_image, top_k=TOP_K_LABELS)

    return cikg_labels, cikg2_labels


print("loading interface")
demo = gr.Interface(
    fn=keywords, 
    inputs=gr.Image(type='pil', label="Image"), 
    outputs=[
        # gr.Label(label="My Algo Image Keywords", num_top_classes=TOP_K_LABELS), 
        gr.Label(label="CI V14 Image Keywords", num_top_classes=TOP_K_LABELS),
        gr.Label(label="CI V32 Image Keywords", num_top_classes=TOP_K_LABELS)
    ]
)

if __name__ == "__main__":
    print("launching interface")
    demo.launch()
