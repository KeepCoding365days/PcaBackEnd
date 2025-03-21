from fastai.vision.all import *
learn=load_learner("resnet_finetuned.pkl")

def classify_images(im):
    pred,idx,probs=learn.predict(im)
    s='Your submitted case has Prostate cancer of IUPC Grade '+pred
    return s