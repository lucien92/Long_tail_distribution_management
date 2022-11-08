from cv2 import imshow
import numpy as np
import keras
import os
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from PIL import Image
from numpy import asarray
import cv2 as cv
from albumentations import (Compose, Rotate, HorizontalFlip, VerticalFlip, Affine, RandomBrightnessContrast, ChannelShuffle, Crop)
import albumentations as A
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# label un dictionnaire qui a chaque image de train associé son label
#list__IDs la liste contenant le chemin vers le simages de train
# AUGMENTATIONS_TRAIN = Compose([
#             Rotate(limit=[0,100], p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             Affine(shear=[-45, 45], p=0.5),
#             RandomBrightnessContrast(p=0.5)
#         ])

AUGMENTATIONS_TRAIN = Compose([
            Crop (x_min=0, y_min=0, x_max=224, y_max=224, always_apply=False, p=1.0)
        ])


class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, labels, path_to_images, cap_sampling, config, n_classes, batch_size, crop, dim=(256,256), n_channels=3, shuffle=True):
        self.path_to_images = path_to_images
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.config = config
        self.batch_size = batch_size
        self.crop = crop
        self.cap_sampling = cap_sampling
        
        
        
        # Group images per species
        self._image_per_specie = {label: [] for label in os.listdir(self.path_to_images)}
        for classe in os.listdir(self.path_to_images):
            for image in os.listdir(os.path.join(self.path_to_images, classe)):
                self._image_per_specie[classe].append(os.path.join(self.path_to_images, classe, image))
        print("sampling activé")
        
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.on_epoch_end()
        return int(np.floor(len(self._images) / self.batch_size)) #renvoie un entier égal au nombre de batch par epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #renvoie les index des images du batch

        # Find list of IDs
        list_IDs_temp = [self._images[k] for k in indexes] #liste contenant le nom des images d'un batch
        

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        
        # for i in range(15):
            
        #     img = Image.fromarray((X[i]*1).astype(np.uint8)).convert('RGB') #avec PIL on doit convertir en RGB
        #     img.show()
            
        X = tf.keras.applications.resnet.preprocess_input(X)
        return X, y
    
    
    #faire fonction qui renvoie les indices des images cappées afin de créer un nouvel indexes (type np.arange(len(self._images))) pour la fonction gettitem
    #et changer list_IDs_temp = [self.list_IDs[k] for k in indexes] en list_IDs_temp = [self._images[k] for k in indexes]
    
    def on_epoch_end(self): 
        
        
        ## Create image set using sampling
        self._images = []

        cap = self.cap_sampling #on aura N images de chaque classe dans notre dataset

        # Initialize counter
        counter = {label: 0 for label in list(os.listdir(self.path_to_images))}
        
        #print("nombre de label verifié:", len(counter))
        
        # Loop to complete each species from the rarest
        counter_min_key = min(counter, key=counter.get)#la classe la plus rare
        counter_min = counter[counter_min_key]#le nombre d'images de cette classe
        while counter_min < cap:
            # Take the first picture and replace it in the queue
            if self._image_per_specie[counter_min_key] != []:
                header_image = self._image_per_specie[counter_min_key].pop(0)#on prend le chemin de la  première image de la classe la plus rare
                self._image_per_specie[counter_min_key].append(header_image)#on la replace dans la queue
            else:
                break

            # Add current image to the image set
            self._images.append(header_image)

            # Increment counters
            counter[counter_min_key] += 1
                
            # Update rarest specie
            counter_min_key = min(counter, key=counter.get)#on ré-actualise la classe la plus rare
            counter_min = counter[counter_min_key]#on ré-écrit le nombre d'images de cette classe
            
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self._images)) #ici on définit self.indexes comme étant la liste des images de train
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
        #on vérifie que le sampling marche bien
        
        labels = list(os.listdir(self.config["data"]["train_images_file"]))
        count = {label: 0 for label in labels}
        for image in self._images:
            for label in labels:
                if label == image.split("/")[-2]:
                    count[label] += 1
                    
        #print("A la fin de cet epoch, on a", count)
        
        
    def parse_image(self, path):
        image = Image.open(path)
        image = image.resize((self.dim))
        image = asarray(image) #on change l'image en array numpy
        return image

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialization
        X = np.empty((self.batch_size, *self.crop, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        lab_to_int = {}
        i = 0
        for label in self.config["data"]["label"]:
            lab_to_int[label] = i
            i += 1
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #on convertit l'image en array)
            img = self.parse_image(os.path.join(self.path_to_images, self.labels[ID], ID)) #image resize en 256*256*3
           
            img = AUGMENTATIONS_TRAIN(image=img)["image"] #crop aléatoire en 224*224*3
            
            # Store sample
            X[i,] = img
            
            # Store class
            #il faut que le label soit un entier
            y[i] = lab_to_int[self.labels[ID]]
            
            
        Y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        
        
        return X, Y
        
    
    