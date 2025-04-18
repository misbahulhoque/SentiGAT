# SentiGAT: A Graph Attention Network-Based Framework for Feature Fusion and Alignment to Enhance Multimodal Sentiment Analysis

A graph attention network (GAT)-based framework for enhanced multimodal sentiment analysis.

## Data and Environment

* Download and extract both MVSA-single and MVSA-multiple datasets from
(https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
* Store the data in the data/ folder.
* data/ folder:
  * 10-fold train/val/test splits (0-indexed) provided in data/ folder for each dataset.
  * valid_pairs.txt contain `file_id, target_label, text_label, image_label` for each valid pair of text-image.
  * `0: Neutral, 1: Positive, and 2: Negative`
* Create an environment using environment.yml with conda.

## Feature Extraction
* Textual feature extraction,
  `python extract/extract_text.py --mvsa single`
* Visual feature extraction (facial expression),
  `python extract_face.py --mvsa single --enable-visual`
* Similarly, extract visual features such as global, object, and image-embedded text.

## Train and Evaluation
* To train and evaluate the model,
 `python models/SentiGAT.py --mvsa single --batch-size 32 --lr 1e-4 --epochs 20 --splits 1 --drop-out 0.5 --hidden-dim 512`
