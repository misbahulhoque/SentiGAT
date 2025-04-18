from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image 
import os, re
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


def get_text_processor(word_stats='twitter', htag=True):
    return TextPreProcessor(
            normalize=['url', 'email', 'phone', 'user'],
            annotate={"hashtag","allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,
            segmenter=word_stats,
            corrector=word_stats,

            unpack_hashtags=htag,
            unpack_contractions=True,
            spell_correct_elong=True,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

def process_tweet(tweet, text_processor):

    proc_tweet = text_processor.pre_process_doc(tweet)

    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]

    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)


class ImageDataset(Dataset):
    def __init__(self, data_dir, img_transform=None, max_size=1024):
        """
        Initialize the dataset.
        :param data_dir: Directory containing the dataset.
        :param img_transform: Optional transform to be applied to the image.
        :param max_size: Maximum dimension for resizing the image.
        """
        self.file_names = pd.read_csv(os.path.join(data_dir, 'valid_pairlist.txt'), header=None)
        self.data_dir = data_dir
        self.img_transform = img_transform
        self.max_size = max_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx, 0])
        img = Image.open(os.path.join(self.data_dir, 'images', fname + '.jpg')).convert('RGB')

        # Resize the image to a fixed size (224x224)
        img = img.resize((224, 224), Image.LANCZOS)

        # Convert to tensor
        original_tensor = transforms.ToTensor()(img)
        if self.img_transform:
            transformed_tensor = self.img_transform(img)
        else:
            transformed_tensor = transforms.ToTensor()(img)

        return original_tensor, transformed_tensor, fname  # Return filename for debugging
    

class TextDataset(Dataset):
    def __init__(self, dloc, txt_transform=None, txt_processor=None):
        self.file_names = pd.read_csv(os.path.join(dloc,'valid_pairlist.txt'), header=None)
        self.dloc = dloc
        self.txt_transform = txt_transform
        self.txt_processor = txt_processor

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx,0])

        text = open(os.path.join(self.dloc, 'texts', fname+'.txt'), 'r', encoding='utf-8', errors='ignore').read().strip().lower()
        
        if self.txt_transform:
            text = self.txt_transform(text, self.txt_processor)

        return text, fname
    

class ImgTxtDataset(Dataset):
    def __init__(self, dloc, txt_transform=None, txt_processor=None, ocr_reader=None):
        self.file_names = pd.read_csv(os.path.join(dloc,'valid_pairlist.txt'), header=None)
        self.dloc = dloc
        self.txt_transform = txt_transform
        self.txt_processor = txt_processor
        self.ocr_reader = ocr_reader

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx,0])

        img_path = os.path.join(self.dloc, fname+'.jpg')

        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)
        # Extract OCR text
        try:
            results = self.ocr_reader.readtext(img_path)
            full_text = " ".join([text for (_, text, _) in results])
            full_text = full_text[:500]
        except OSError as e:
            print(f"Skipping {img_path} due to error: {e}")
            full_text = ""
        return full_text, fname
