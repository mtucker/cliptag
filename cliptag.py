import torch
import requests
import open_clip
from PIL import Image
from io import BytesIO
import os


def load_keywords_from_files(directory):
    keyword_list = []
    for filename in os.listdir(directory):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                # Read lines and strip leading/trailing whitespace
                keywords = [line.strip() for line in file]
                # Add keywords to the master keyword_list
                keyword_list.extend(keywords)
    print(f"Loaded {len(keyword_list)} keywords...")
    return keyword_list

def encode_keywords(candidate_keywords, model, device, batch_size=1000):
    # Tokenize and encode the candidate keywords in batches
    print("Tokenizing and encoding candidate keywords...")
    text_features = []
    for i in range(0, len(candidate_keywords), batch_size):
        print(f"Batch {i+1}...")
        batch_keywords = candidate_keywords[i:i + batch_size]
        batch_tokens = open_clip.tokenize(batch_keywords).to(device)
        with torch.no_grad():
            batch_features = model.encode_text(batch_tokens)
        text_features.append(batch_features)
    # Concatenate the encoded features from all batches
    text_features = torch.cat(text_features, dim=0)

    return text_features


def generate_keywords_for_image(image, candidate_keywords, text_features, model, preprocess, device, top_k=5, batch_size=100):

  try:
    print("Preprocessing image...")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Encode the image and keywords
    print("Encoding image and keywords...")
    with torch.no_grad():
        print("Encoding image...")
        image_features = model.encode_image(image_tensor)

    # Calculate the similarity scores
    print("Calculating similarity scores...")
    similarities = torch.matmul(image_features, text_features.T)

    # Get the top K keywords based on the similarity scores
    print("Getting top keywords...")

    print(similarities)
    values, indices = torch.topk(similarities, top_k, dim=-1)

    values_list = values.squeeze(0).tolist()
    indices_list = indices.squeeze(0).tolist()

    top_keywords = {candidate_keywords[idx]: values_list[i] / 100 for i, idx in enumerate(indices_list)}
    return top_keywords

  except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL: {e}")
        return []