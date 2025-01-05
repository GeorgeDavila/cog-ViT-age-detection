from typing import List

import torch
from PIL import Image
from functools import partial
from cog import BasePredictor, BaseModel, Input, Path
import requests
from io import BytesIO
import numpy as np

#from transformers import ViTFeatureExtractor, ViTForImageClassification #deprecated left in for reference 
from transformers import AutoImageProcessor, AutoModelForImageClassification

id2label = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+",
    }

MODEL = "nateraw/vit-age-classifier"
MODEL_CACHE = "model-cache"
device = "cuda"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL, cache_dir=MODEL_CACHE, local_files_only=True,
        ).to(device)

        self.transforms = AutoImageProcessor.from_pretrained(
            MODEL, cache_dir=MODEL_CACHE, local_files_only=True,
            use_fast=True
            ).to(device)

    def predict(
        self,
        image: Path = Input(
            description="Image of 1 person",
        ),
        child_confidence_threshold: float = Input(
            description="The confidence threshold for sum of the child-containing classes.",
            default= 0.04
        ),
    ) -> List[str]:
        """Run the provided image age detection"""

        im = Image.open(image).convert('RGB')
        inputs = self.transforms(im, return_tensors='pt')
        output = self.model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Predicted Classes
        preds = proba.argmax(1)

        likeliest_age_confidence_score = round(max(proba[0]).item(), 2)
        likeliest_age = id2label[int(preds)]

        all_ages_confidence_scores_list = [i.item() for i in proba[0]]
        all_ages_confidence_scores_dict = { id2label[i] : all_ages_confidence_scores_list[i] for i in range(len(all_ages_confidence_scores_list))}

        print(all_ages_confidence_scores_dict)

        isThisChildBool = (np.sum(all_ages_confidence_scores_list[0:3]) > child_confidence_threshold)
        print("isThisChildBool == ", isThisChildBool)

        print("likeliest_age class ==", likeliest_age, "  likeliest_age_confidence_score == ", likeliest_age_confidence_score)

        return likeliest_age, isThisChildBool, likeliest_age_confidence_score, all_ages_confidence_scores_dict