import math
from typing import List
import torch
import open_clip
import os
from clip_interrogator import Config, Interrogator as CIInterrogator, LabelTable, load_list
from tqdm import tqdm

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

class Interrogator(CIInterrogator):
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.caption_offloaded = True
        self.clip_offloaded = True
        self.load_clip_model()

    def load_clip_model(self):
        config = self.config

        clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)

        if config.clip_model is None:
            if not config.quiet:
                print(f"Loading CLIP model {config.clip_model_name}...")

            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=clip_model_pretrained_name, 
                precision='fp16' if config.device == 'cuda' else 'fp32',
                device=config.device,
                jit=False,
                cache_dir=config.clip_model_path
            )
            self.clip_model.eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

class ImageKeywordGenerator:

    def __init__(self, features_file_dir, config=None):
        if not config:
            config = Config(caption_model_name=None)
        self.ci = Interrogator(config)
        cache_desc = features_file_dir.replace('/', '_')
        self.table = LabelTable(load_keywords_from_files(features_file_dir), cache_desc, self.ci)
        self.device = self.table.device
        self.chunk_size = self.table.chunk_size
        self.config = self.table.config
        self.labels = self.table.labels
        self.embeds = self.table.embeds

    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1, reverse: bool=False) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
            if reverse:
                similarity = -similarity

        values, indices = torch.topk(similarity, top_count, dim=-1)

        values_list = values.squeeze(0).tolist()
        indices_list = indices.squeeze(0).tolist()
        
        return [(values_list[i], indices_list[i]) for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=1, reverse: bool=False) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count, reverse=reverse)
            return {self.labels[i]: val for val, i in tops}

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk, reverse=reverse)
            top_labels.extend([self.labels[start+i] for _, i in tops])
            top_embeds.extend([self.embeds[start+i] for _, i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return {self.labels[i]: val for val, i in tops}


    def generate_keywords_for_image(self, image, top_k=5):
        return self.rank(self.ci.image_to_features(image), top_count=top_k)