import requests
from PIL import Image
from io import BytesIO
import numpy as np

#from transformers import ViTFeatureExtractor, ViTForImageClassification #deprecated left in for reference 
from transformers import AutoImageProcessor, AutoModelForImageClassification

#We restructure this in predict.py as we want to load model in the cog container setup 

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

def getAgeFromImage(imageSource:str, under18Threshold=0.04, isFromURL=False):
    '''
    under18Threshold is the max allowed value for the sum of the 0-19 classes. Default is 4% aka model is 96% confident the image is 20 or older
    '''
    # Get example image from official fairface repo + read it in as an image
    if isFromURL:
        r = requests.get(imageSource)
        im = Image.open(BytesIO(r.content)).convert('RGB')
    else:
        im = Image.open(imageSource).convert('RGB')

    # Init model, transforms
    model = AutoModelForImageClassification.from_pretrained('nateraw/vit-age-classifier') #deprecated left in for reference
    transforms = AutoImageProcessor.from_pretrained('nateraw/vit-age-classifier', use_fast=True)
    #model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier') #deprecated left in for reference
    #transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier') #deprecated left in for reference

    # Transform our image and pass it through the model
    inputs = transforms(im, return_tensors='pt')
    output = model(**inputs)

    print(output)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)
    print(proba[0])

    # Predicted Classes
    preds = proba.argmax(1)

    likeliest_age_confidence_score = round(max(proba[0]).item(), 2)
    likeliest_age = id2label[int(preds)]

    all_ages_confidence_scores_list = [i.item() for i in proba[0]]
    all_ages_confidence_scores_dict = { id2label[i] : all_ages_confidence_scores_list[i] for i in range(len(all_ages_confidence_scores_list))}

    print(all_ages_confidence_scores_dict)

    print(all_ages_confidence_scores_list[0:3])
    print( np.sum(all_ages_confidence_scores_list[0:3]) )

    isThisChildBool = (np.sum(all_ages_confidence_scores_list[0:3]) > under18Threshold)
    print(isThisChildBool)

    print(likeliest_age, likeliest_age_confidence_score)

    return likeliest_age, isThisChildBool, likeliest_age_confidence_score, all_ages_confidence_scores_dict

getAgeFromImage('testimg2.jpg')