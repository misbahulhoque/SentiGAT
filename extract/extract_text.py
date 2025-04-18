import argparse
import torch
import torch
import clip
from PIL import Image
import numpy as np
import os
import json
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from helper import TextDataset, get_text_processor, process_tweet
import matplotlib.pyplot as plt

# Basic initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

txt_processor = get_text_processor(htag=True)
txt_transform = process_tweet

def extract_sentence_embedding(text):
    text_tokens = clip.tokenize([text]).to(device)              # Shape: [1, 77]

    with torch.no_grad():
        sentence_embedding = model.encode_text(text_tokens)     # Shape: [1, 512]
        
    return sentence_embedding.squeeze(0).cpu().numpy().tolist()

def extract_word_embeddings(text):
    words = text.split()
    word_embeddings = []
    
    # Extract embeddings for each word
    for word in words:
        word_tokens = clip.tokenize([word]).to(device)          # Shape: [1, 77]
        with torch.no_grad():
            word_embedding = model.encode_text(word_tokens)     # Shape: [1, 512]
        word_embeddings.append(word_embedding.squeeze(0).cpu().numpy().tolist())
    
    return word_embeddings

def extract_embeddings(mvsa='single'):
    dloc = f'../data/mvsa_{mvsa}/'
    sentence_embeddings = []
    word_embeddings = []

    dataset = TextDataset(dloc, txt_transform=txt_transform, txt_processor=txt_processor) 
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(f"Processing batch {i}")

        txt_inps, fnames = batch
        for text, fname in zip(txt_inps, fnames):

            sentence_embedding = extract_sentence_embedding(text)
            sentence_embeddings.append(sentence_embedding)
        
            word_embedding = extract_word_embeddings(text)
            word_embeddings.append(word_embedding)

    with open(f'./features/texts_{mvsa}.json', 'w') as f:
        json.dump({
            'sentence_embeddings': sentence_embeddings,
            'word_embeddings': word_embeddings
        }, f)

    return sentence_embeddings, word_embeddings

def main():
    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    args = parser.parse_args()

    sentence_embeddings, word_embeddings = extract_embeddings(mvsa=args.mvsa)
    print("Sentence Embeddings Shape:", np.array(sentence_embeddings).shape)
    print("Word Embeddings Shape:", np.array(word_embeddings).shape)

if __name__ == '__main__':
    main()