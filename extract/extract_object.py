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
from helper import ImageDataset
import matplotlib.pyplot as plt
from ultralytics import YOLO  # YOLOv8 for object detection

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Basic initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),          
])

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")     # Load YOLOv8 Nano model
object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s").to(device)

def detect_objects(image):
        """
        Detect objects in the image using YOLOv5.
        :param image: Input image (PIL Image).
        :return: A list of tuples: [(x1, y1, x2, y2, "object: <label>"), ...].
        """
        objects = []
        image_np = np.array(image)

        # Detect objects using YOLOv5
        object_results = object_detector(image_np)

        for _, row in object_results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            label = row["name"]
            confidence = float(row["confidence"])
            objects.append((label, confidence, (x1, y1, x2, y2)))

        return objects

def extract_object_embeddings(image, objects):
    object_embeddings = []

    for label, confidence, (x1, y1, x2, y2) in objects:
        if label != "person":  # Filter out "person" objects
            # Crop the object region
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = preprocess(cropped_image).unsqueeze(0).to(device)  # Preprocess for CLIP

            # Extract CLIP embedding for the object
            with torch.no_grad():
                object_embedding = model.encode_image(cropped_image)  # Shape: [1, 512]
            object_embeddings.append(object_embedding)

    # If no relevant objects are found, return a zero vector
    if not object_embeddings:
        return [torch.zeros(1, 512).to(device)]  # Return a list with a single zero vector

    return object_embeddings

def visualize(image, objects, id, output_folder="./check_objects/"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert the image to a numpy array
    image_np = np.array(image)

    # If no objects are detected, add a message to the image
    if not objects:
        plt.figure()
        plt.imshow(image_np)
        plt.text(10, 10, "No object detected", color='r', fontsize=12, backgroundcolor='white')
        plt.axis('off')
        full_output_path = os.path.join(output_folder, f"full_image_{id}.png")
        plt.savefig(full_output_path)
        plt.close()
        print(f"Saved full image to {full_output_path}")
        return

    # Create a new figure for the image
    plt.figure()
    plt.imshow(image)
    for obj in objects:
        label, confidence, region = obj  # Unpack the object tuple
        x1, y1, x2, y2 = region  # Unpack the bounding box coordinates
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none'))
        plt.text(x1, y1 - 10, f"Label: {label} Conf: {confidence:.2f}", color='r', fontsize=12, backgroundcolor='white')
    plt.axis('off')

    # Save the plot to a file
    full_output_path = os.path.join(output_folder, f"full_image_{id}.png")
    plt.savefig(full_output_path)
    plt.close()  # Close the figure to free up memory
    print(f"Saved full image to {full_output_path}")   

def extract_embeddings(mvsa='single', enable_visual=False):
    dloc = f'../data/mvsa_{mvsa}/'
    object_embeddings_all = []  # List to store embeddings per image

    # Load the dataset with img_transforms and max_size
    dataset = ImageDataset(dloc, img_transform=img_transforms, max_size=1024)  # Apply transforms and resize
    dt_loader = DataLoader(dataset, batch_size=128, sampler=SequentialSampler(dataset))  # Increase batch size

    for i, batch in enumerate(dt_loader):
        print(f"Processing batch {i}")

        # Unpack the batch into original and transformed tensors
        original_tensor, _, fname = batch

        for j in range(len(original_tensor)):  # Iterate over the batch
            # Convert tensor to PIL Image
            original_image = transforms.ToPILImage()(original_tensor[j].cpu())

            # Detect objects in the image
            objects = detect_objects(original_image)

            # Extract CLIP embeddings for relevant objects
            object_embeddings = extract_object_embeddings(original_image, objects)
            object_embeddings = [embed.cpu().numpy().tolist()[0] for embed in object_embeddings]  # Convert to list and remove extra nesting

            # Append object embeddings for this image to the main list
            object_embeddings_all.append(object_embeddings)

            # Optional: Visualize detected objects
            if enable_visual:
                visualize(original_image, objects, id=f"{i}_{j}")

    # Save object embeddings to JSON
    with open(f'./features/object_{mvsa}.json', 'w') as f:
        json.dump({'feats': object_embeddings_all}, f)

    return object_embeddings_all

# Modify main execution block
def main():
    parser = argparse.ArgumentParser(description='Extract object features')
    parser.add_argument('--mvsa', type=str, default='single', choices=['single', 'multiple'])
    parser.add_argument('--enable-visual', default=False)
    args = parser.parse_args()

    object_embedding = extract_embeddings(
        mvsa=args.mvsa,
        enable_visual=args.enable_visual,
    )
    print(np.array(object_embedding).shape)

if __name__ == '__main__':
    main()