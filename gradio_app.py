import gradio as gr
import torch 
from cliptag.cliptag import CIImageKeywordGenerator
from clip_interrogator import Config

device = None

FEATURES_FILE_DIR = "features/photos"
TOP_K_LABELS = 100
CLIP_MODELS = [
    "ViT-L-14/openai",
    "ViT-B-32/laion2b_s34b_b79k"
]

print("loading model")
if torch.cuda.is_available():
    # Use the GPU (CUDA) as the device
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    # Use the CPU as the device
    device = torch.device("cpu")
    print("Using CPU")

keyword_generators = []
outputs = []

for clip_model_name in CLIP_MODELS:
    config = Config(clip_model_name=clip_model_name, caption_model_name=None, device=device)
    keyword_generators.append(CIImageKeywordGenerator(FEATURES_FILE_DIR, config))
    outputs.append(gr.Label(label=f"{clip_model_name} Image Keywords", num_top_classes=TOP_K_LABELS))

def keywords(input_image):
    labels = []

    for keyword_generator in keyword_generators:
        labels.append(keyword_generator.generate_keywords_for_image(input_image, top_k=TOP_K_LABELS))

    return labels

print("loading interface")
demo = gr.Interface(
    fn=keywords, 
    inputs=gr.Image(type='pil', label="Image"), 
    outputs=outputs
)

if __name__ == "__main__":
    print("launching interface")
    demo.launch()
