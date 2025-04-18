import argparse
import torch
import torch
import clip
import numpy as np
import os
import json
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision import transforms
from helper import ImgTxtDataset, get_text_processor, process_tweet
import matplotlib.pyplot as plt
import logging

import easyocr
gpu_available = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=gpu_available)


# Basic initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

txt_processor = get_text_processor(htag=True)
txt_transform = process_tweet

def extract_sentence_embedding(text):
    if not text or text.strip() == "":
        logging.warning("Empty or None text encountered. Returning zero vector.")
        return [0.0] * 512  # Return a 512D zero vector for empty or None text

    # Truncate text if it exceeds CLIP's token limit (77 tokens)
    max_tokens = 77
    text_tokens = clip.tokenize([text], truncate=True).to(device)  # Truncate if necessary

    with torch.no_grad():
        sentence_embedding = model.encode_text(text_tokens)  # Shape: [1, 512]

    return sentence_embedding.squeeze(0).cpu().numpy().tolist()

def extract_embeddings(mvsa='single'):
    dloc = f'../data/mvsa_{mvsa}/'
    sentence_embeddings = []

    dataset = ImgTxtDataset(dloc, txt_transform=txt_transform, txt_processor=txt_processor, ocr_reader=reader) 
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(f"Processing batch {i}")

        txt_inps, fnames = batch
        for text, fname in zip(txt_inps, fnames):
            sentence_embedding = extract_sentence_embedding(text)
            sentence_embeddings.append(sentence_embedding)

    with open(f'./features/imgtxt_{mvsa}.json', 'w') as f:
        json.dump({
            'sentence_embeddings': sentence_embeddings
        }, f)

    return sentence_embeddings


def main():
    parser = argparse.ArgumentParser(description='Extract image text features')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    args = parser.parse_args()

    sentence_embeddings = extract_embeddings(mvsa=args.mvsa)
    print("Image-Text Embeddings Shape:", np.array(sentence_embeddings).shape)

if __name__ == '__main__':
    main()