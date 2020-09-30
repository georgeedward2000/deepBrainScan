from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback
import os
import PIL
from PIL import Image
import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

pathModel = os.path.join(settings.BASE_DIR, 'deepBrainScan/models/unet_brain_mri_seg.hdf5')
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
with session.graph.as_default():
    K.set_session(session)
    model = tf.keras.models.load_model(pathModel, compile=False)



# img = cv2.resize(img ,(im_height, im_width))
#     img = img / 255
#     img = img[np.newaxis, :, :, :]
#     pred=model.predict(img)

def index(request, *args, **kargs):

    if  request.method == "POST":
        f = request.FILES['sentFile'] # here you get the files needed
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        file_url = str(settings.BASE_DIR) + file_url
        original = load_img(file_url, target_size=(256, 256))

        plt.imshow(np.squeeze(original))
        plt.title("Original:")
        plt.savefig('original.png')

        # print("AICI")
        # print(original)
        original = img_to_array(original)
        
        img = cv2.resize(original ,(256, 256))
        original = original / 255
        processedImage = original[np.newaxis, :, :, :]
        # print(processedImage.shape)

       
       # launch graph in session
        with graph.as_default():
            K.set_session(session)
            predictions = model.predict(processedImage)

        # print(predictions)

        plt.imshow(np.squeeze(predictions) > 0.5, cmap='jet', alpha=0.5)
        plt.title("Detection:")
       # plt.imshow(predictions)
        plt.savefig('pred.png')
        
        file_PRED= default_storage.url('testPred3.png')
        # print("AICI2")
        # print(file_PRED)

        images = [Image.open(x) for x in ['original.png', 'pred.png']]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save('trifecta.png')
        trif = str(settings.BASE_DIR) + '/trifecta.png'
        with open(trif, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        response["name"] = image_data
        
        # print("AICI TRIF :")
        # print(trif)
        #response['name'] = trif
        return render(request,'homepage.html',response)
    else:
        return render(request,'homepage.html')