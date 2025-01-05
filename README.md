# cog-ViT-age-detection

The variable under18Threshold is the max allowed value for the sum of the 0-19 classes. Default is 4% aka model is 96% confident the image is 20 or older. We use 4% threshold here as 2 standard deviations ie 95.4% confidence is [commonly used in medicine](https://www.jospt.org/doi/10.2519/jospt.2019.0706) so its probably good enough for your use case. Adjust as needed.


## References
    - https://huggingface.co/nateraw/vit-age-classifier
    - https://huggingface.co/spaces/Paresh/Facial-feature-detector/blob/main/src/face_demographics.py
    - https://huggingface.co/spaces/dennisjooo/Age-and-Emotion-Classifier/blob/main/app.py