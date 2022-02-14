import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True)
args = parser.parse_args()
class_names =  ["INGLES", "MTG","MYL","YUGIOH"]

def load_and_prep_image(filename,img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img,size=[img_shape,img_shape])
    img = img/255.
    return img

def pred_and_plot(model,filename,class_names=class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img,axis=0))
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} {round(max(pred[0])*100)}%")
    plt.axis(False)
    plt.show()

def main(filename):
    
    modelo = tf.keras.models.load_model("Model_0_cards")
    pred_and_plot(modelo,filename,class_names=class_names)



if __name__ == "__main__":
    main(args.filepath)