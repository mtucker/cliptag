import concurrent.futures
import gradio as gr
import os
import torch 
from cliptag.cliptag import ImageKeywordGenerator
from cliptag.loaders import load_clip_generators, load_device
from clip_interrogator import Config


FEATURES_FILE_DIR = "features/photos"
CACHE_DIR = "cache"
TOP_K_LABELS = 100
CLIP_MODELS = [
    "ViT-L-14/openai",
    "ViT-L-14/laion2b_s32b_b82k",
    "ViT-B-32/laion2b_s34b_b79k"
]

device = load_device()
keyword_generators = load_clip_generators(CLIP_MODELS, FEATURES_FILE_DIR, device, CACHE_DIR)

def generate_keywords(keyword_generator, input_image, top_k):
    return keyword_generator.generate_keywords_for_image(input_image, top_k=top_k)

def keywords(input_image):
    labels = []

    # Use ThreadPoolExecutor to parallelize keyword generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a lambda to pass the required arguments to the generate_keywords function
        tasks = [executor.submit(generate_keywords, kg, input_image, TOP_K_LABELS) for kg in keyword_generators]

        # Collect the results
        for future in concurrent.futures.as_completed(tasks):
            labels.append(future.result())

    return labels

print("Loading interfacez...")
outputs = []
examples = [os.path.join("examples", example) for example in os.listdir("examples")]

with gr.Blocks() as demo:
    gr.Markdown("Upload an image below to generate keywords using different CLIP model versions.")
    inp = gr.Image(type='pil', label="Image")
    btn = gr.Button("Generate Keywords")
    gr.Examples(
        examples=examples,
        inputs=inp,
    )
    with gr.Row():
        for clip_model_name in CLIP_MODELS:
            output = gr.Label(label=f"{clip_model_name} Image Keywords", num_top_classes=TOP_K_LABELS)
            outputs.append(output)

    btn.click(fn=keywords, inputs=inp, outputs=outputs)

if __name__ == "__main__":
    print("Launching interface...")
    demo.launch()
