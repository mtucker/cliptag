import os
import pickle

import torch
from cliptag import ImageKeywordGenerator
from clip_interrogator import Config


def load_device():
    device = None

    if torch.cuda.is_available():
        # Use the GPU (CUDA) as the device
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        # Use the CPU as the device
        device = torch.device("cpu")
        print("Using CPU")

    return device


def load_clip_generators(clip_model_names, feature_files_dir, device, cache_dir):
    """
    Load models from file or create new ones
    :return: list of ImageKeywordGenerator objects
    """
    generator_cache_dir = f"{cache_dir}/generators"
    embeds_cache_dir = f"{cache_dir}/embeds"
    pickled_generators = f"{generator_cache_dir}/keyword_generators.pkl"

    if not os.path.exists(generator_cache_dir):
        os.makedirs(generator_cache_dir)

    if os.path.exists(pickled_generators):
        print("Loading models from file")
        with open(pickled_generators, "rb") as f:
            keyword_generators = pickle.load(f)
    else:
        keyword_generators = []

        for clip_model_name in clip_model_names:
            config = Config(
                clip_model_name=clip_model_name,
                caption_model_name=None,
                device=device,
                cache_path=embeds_cache_dir,
            )
            keyword_generators.append(ImageKeywordGenerator(feature_files_dir, config))

        with open(pickled_generators, "wb") as f:
            pickle.dump(keyword_generators, f)

    return keyword_generators
