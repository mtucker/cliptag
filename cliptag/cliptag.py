import torch
import open_clip
import os


class ImageKeywordGenerator:
    def __init__(self, model, preprocess, device, features_file_dir):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self._candidate_keyword_labels = self._load_keywords_from_files(features_file_dir)
        self._candidate_keyword_features = self._encode_keywords(self._candidate_keyword_labels)

    def _load_keywords_from_files(self, directory):
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

    def _encode_keywords(self, candidate_keywords, batch_size=1000):
        # Tokenize and encode the candidate keywords in batches
        print("Tokenizing and encoding candidate keywords...")
        text_features = []
        for i in range(0, len(candidate_keywords), batch_size):
            print(f"Batch {i+1}...")
            batch_tokens = open_clip.tokenize(candidate_keywords[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                batch_features = self.model.encode_text(batch_tokens)
            text_features.append(batch_features)
        # Concatenate the encoded features from all batches
        text_features = torch.cat(text_features, dim=0)
        return text_features

    def generate_keywords_for_image(self, image, top_k=5):
        try:
            print("Preprocessing image...")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Encode the image and keywords
            print("Encoding image and keywords...")
            with torch.no_grad():
                print("Encoding image...")
                image_features = self.model.encode_image(image_tensor)

            # Calculate the similarity scores
            print("Calculating similarity scores...")
            similarities = torch.matmul(image_features, self._candidate_keyword_features.T)

            # Get the top K keywords based on the similarity scores
            print("Getting top keywords...")
            values, indices = torch.topk(similarities, top_k, dim=-1)

            values_list = values.squeeze(0).tolist()
            indices_list = indices.squeeze(0).tolist()

            top_keywords = {self._candidate_keyword_labels[idx]: values_list[i] / 100 for i, idx in enumerate(indices_list)}
            return top_keywords

        except Exception as e:
            print(f"Error generating keywords for image: {e}")
            return []
