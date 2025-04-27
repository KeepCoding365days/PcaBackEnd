import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
def predict_grade(img_path):
    model=tf.keras.layers.TFSMLayer('patch_classifer.keras')
    
    img_path = "/kaggle/input/pca-patches-classification/TrainingData/0/00a7fb880dc12c5de82df39b30533da9_26_5.png"  # Change to your image path
    img = load_img(img_path, target_size=(256, 256))        # resize image

    # Convert to array
    img_array = img_to_array(img)

    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Optionally: normalize pixels if your model expects it
    #img_array = img_array / 255.0
    predictions=model(img_array)
    return tf.argmax(predictions, axis=-1).numpy()[0]