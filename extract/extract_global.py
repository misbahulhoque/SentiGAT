import argparse
import torch
import torch
import clip
import numpy as np
import json
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from helper import ImageDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Basic initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

def extract_embeddings(mvsa='single'):
    dloc = f'../data/mvsa_{mvsa}/'
    global_embeddings= []           # List to store embeddings per image

    dataset = ImageDataset(dloc, img_transform=img_transforms, max_size=1024)
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(f"Processing batch {i}")
        images, _ = batch 

        # Process the entire batch at once
        with torch.no_grad():
            image_features = model.encode_image(images.to(device))  # Shape: [batch_size, 512]
            global_embeddings.extend(image_features.cpu().numpy().tolist())


    with open(f"./features/{mvsa}_global.json", "w") as f:
        json.dump({"feats": global_embeddings}, f)

    return global_embeddings


def main():
    parser = argparse.ArgumentParser(description='Extract global features')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    args = parser.parse_args()

    global_embeddings = extract_embeddings(
        mvsa=args.mvsa,
    )
    print(np.array(global_embeddings).shape)

if __name__ == '__main__':
    main()