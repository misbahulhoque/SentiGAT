import numpy as np
import json
import torch
from typing import List, Dict, Any
import torch.nn.functional as F

floc = '../extract/features/'

def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def to_tensor(emb):
    return torch.tensor(emb, dtype=torch.float32) if not isinstance(emb, torch.Tensor) else emb

def get_features(mvsa='single'):
    
    # Load textual features
    texts_data = load_json(f"{floc}texts_{mvsa}.json")
    sentence_emb = np.array(texts_data.get("sentence_embeddings", []))
    word_emb = [np.array(words) for words in texts_data.get("word_embeddings", [])]
    
    # Load visual features
    global_data = load_json(f"{floc}global_{mvsa}.json")
    face_data = load_json(f"{floc}faces_{mvsa}.json")
    object_data = load_json(f"{floc}object_{mvsa}.json")
    imgtxt_data = load_json(f"{floc}imgtxt_{mvsa}.json")

    global_emb = np.array(global_data.get("feats", []))
    face_emb = np.array(face_data.get("feats", []))
    object_emb = [np.array(obj) for obj in object_data.get("feats", [])]
    imgtxt_emb = np.array(imgtxt_data.get("sentence_embeddings", []))
    return sentence_emb, word_emb, global_emb, face_emb, object_emb, imgtxt_emb