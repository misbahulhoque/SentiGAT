import argparse
import torch
import clip
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
import os
import json
from torch.utils.data import DataLoader, SequentialSampler
from deepface import DeepFace
import matplotlib.pyplot as plt
from helper import ImageDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Basic initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model_name = "ViT-B/32"
face_detector = MTCNN(keep_all=True, device=device)
model, preprocess = clip.load(clip_model_name, device=device)

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),          # Convert to tensor
])


def detect_faces(image):
    """
    Detect faces in the image using facenet MTCNN.
    :param image: Input image (PIL Image).
    :return: A list of tuples: [(x1, y1, x2, y2), ...].
    """
    image_np = np.array(image)
    boxes, _ = face_detector.detect(image_np)
    face_regions = []
    if boxes is not None:  # Check if any faces were detected
        for box in boxes:
            x1, y1, x2, y2 = box
            face_regions.append((int(x1), int(y1), int(x2), int(y2)))
    return face_regions

def extract_text_embedding(text):
    """
    Extract CLIP text embedding for a given text using the `clip` library.
    :param text: Input text (str).
    :param model: Loaded CLIP model.
    :param device: Device (e.g., "cuda" or "cpu").
    :return: CLIP text embedding (torch.Tensor) of shape [1, 512].
    """
    # Tokenize the input text
    text_tokens = clip.tokenize([text]).to(device)  # Shape: [1, 77]

    # Encode the text into CLIP embedding space
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)  # Shape: [1, 512]

    return text_embedding

def get_facial_expression_embedding(image):
    """
    Extracts facial expression embeddings from an image using DeepFace and maps them to CLIP space.
    :param image: Input PIL image.
    :return: Weighted emotion embedding in CLIP space (torch.Tensor) of shape [1, 512].
    """
    # Detect faces in the image
    face_regions = detect_faces(image)
    if not face_regions:
        return torch.zeros(1, 512).to(device)  # Zero vector if no faces are detected

    # Detect facial expressions using DeepFace
    emotion_probs_list = []
    for region in face_regions:
        x1, y1, x2, y2 = region
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image_np = np.array(cropped_image)

        try:
            results = DeepFace.analyze(cropped_image_np, actions=["emotion"], enforce_detection=False)
            if results:
                # Extract emotion probabilities
                emotion_probs = results[0]["emotion"]
                
                emotion_probs_list.append(emotion_probs)
        except Exception as e:
            print(f"Error detecting facial expressions: {e}")

    if not emotion_probs_list:
        return torch.zeros(1, 512).to(device)  # Return zero vector if no face detected

    # Average emotion probabilities across all detected faces
    avg_emotion_probs = {
        "angry": np.mean([ep["angry"] for ep in emotion_probs_list]),
        "disgust": np.mean([ep["disgust"] for ep in emotion_probs_list]),
        "fear": np.mean([ep["fear"] for ep in emotion_probs_list]),
        "happy": np.mean([ep["happy"] for ep in emotion_probs_list]),
        "sad": np.mean([ep["sad"] for ep in emotion_probs_list]),
        "surprise": np.mean([ep["surprise"] for ep in emotion_probs_list]),
        "neutral": np.mean([ep["neutral"] for ep in emotion_probs_list]),
    }


    # Map emotions to CLIP space using emotion words
    emotion_words = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion_embeddings = [extract_text_embedding(word) for word in emotion_words]  # List of [1, 512] tensors
    emotion_embeddings = torch.stack(emotion_embeddings).squeeze(1)  # Shape: [7, 512]

    # Weighted sum of emotion embeddings
    weights = torch.tensor(list(avg_emotion_probs.values())).unsqueeze(1).to(device)  # Shape: [7, 1]
    
    weighted_embedding = torch.sum(emotion_embeddings * weights, dim=0).unsqueeze(0)  # Shape: [1, 512]


    return weighted_embedding

def get_visual_info(image):
    face_regions = detect_faces(image)
    expressions = []
    for region in face_regions:
        x1, y1, x2, y2 = region
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image_np = np.array(cropped_image)

        try:
            results = DeepFace.analyze(cropped_image_np, actions=["emotion"], enforce_detection=False)
            if results:
                    expression = results[0]["dominant_emotion"]
                    expressions.append(expression)
        except Exception as e:
            print(f"Error detecting facial expressions: {e}")
    return face_regions, expressions

def visualize(image, face_regions, expressions, id, output_folder="./check_result/"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert the image to a numpy array
    image_np = np.array(image)

    # If no regions are detected, add a message to the image
    if not face_regions:
        plt.figure()
        plt.imshow(image_np)
        plt.text(10, 10, "No faces detected", color='r', fontsize=12, backgroundcolor='white')
        plt.axis('off')
        full_output_path = os.path.join(output_folder, f"full_image_{id}.png")
        plt.savefig(full_output_path)
        plt.close()
        print(f"Saved full image to {full_output_path}")
        return

    # Create a new figure for the image
    plt.figure()
    plt.imshow(image)
    for region, expression in zip(face_regions, expressions):
        x1, y1, x2, y2 = region
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none'))
        plt.text(x1, y1 - 10, f"Expression: {expression}", color='r', fontsize=12, backgroundcolor='white')
    plt.axis('off')

    # Save the plot to a file
    full_output_path = os.path.join(output_folder, f"full_image_{id}.png")
    plt.savefig(full_output_path)
    plt.close()  # Close the figure to free up memory
    print(f"Saved full image to {full_output_path}")


def extract_embeddings(mvsa='single', enable_visual=False):
    dloc = f'../data/mvsa_{mvsa}/'
    facial_expression_feats = []

    # Load the dataset with img_transforms and max_size
    dataset = ImageDataset(dloc, img_transform=img_transforms, max_size=1024)  # Apply transforms and resize
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))

    for i, batch in enumerate(dt_loader):
        print(f"Processing batch {i}")

        # Unpack the batch into original and transformed tensors
        original_tensor, _, fname = batch

        for j in range(len(original_tensor)):  # Iterate over the batch
            # Convert tensor to PIL Image
            original_image = transforms.ToPILImage()(original_tensor[j].cpu())

            # Detect facial expressions on the original image
            facial_expression_embedding = get_facial_expression_embedding(original_image)

            # Convert embedding to a flat list and append to the main list
            facial_expression_feats.append(facial_expression_embedding.cpu().numpy().tolist()[0])  # Remove extra nesting

            # Optional: Visualize detected expressions
            if enable_visual:
                face_regions, expressions = get_visual_info(original_image)
                visualize(original_image, face_regions, expressions, id=f"{i}_{j}")

    # Save facial expression embeddings to JSON
    with open(f'./features/faces_{mvsa}.json', 'w') as f:
        json.dump({'feats': facial_expression_feats}, f)

    return facial_expression_feats

def main():
    parser = argparse.ArgumentParser(description='Extract facial expression features')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    parser.add_argument('--enable-visual', default=False)
    args = parser.parse_args()

    face_expression = extract_embeddings(
        mvsa=args.mvsa,
        enable_visual=args.enable_visual
    )
    print(np.array(face_expression).shape)

if __name__ == '__main__':
    main()