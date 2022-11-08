## Téléchargement de la base de données
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import argparse
import os
from BatchGenerator import DataGenerator
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
import keras.backend as K
import tensorflow as tf

#lien au json

argparser = argparse.ArgumentParser(
    description='Train and validate ResNet model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='//home/acarlier/code/code_to_present_class_imbalanced_issue/config/plantnet_val_sampling.json',
    help='path to configuration file')


def _main_(args):
    config_path = args.conf
    import tensorflow as tf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    
        
    IMG_SIZE_train = 256
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config["data"]["train_images_file"],
        labels='inferred',
        label_mode='categorical',
        shuffle = False,
        batch_size=config["train"]["batch_size"],
        image_size=(IMG_SIZE_train, IMG_SIZE_train))
    class_names = train_ds.class_names
    print(class_names)
    nb_classes = len(class_names)
    ## Chargement des données
    from tensorflow import keras
    from tensorflow.keras import layers
    # Paramètres
    IMG_SIZE_train = 256 # pour utiliser ResNet
    IMG_SIZE_val = 224
    # Récupération des dataset pour l'entraînement (train, val)
    # Shuffle à false pour avoir accès aux images depuis
    # leur chemin d'accès avec train_ds.file_paths
    train_ds = keras.utils.image_dataset_from_directory(
        directory=config["data"]["train_images_file"],
        labels='inferred',
        label_mode='categorical',
        shuffle = False,
        batch_size=config["train"]["batch_size"],
        image_size=(IMG_SIZE_train, IMG_SIZE_train))

    validation_ds = keras.utils.image_dataset_from_directory(
        directory=config["data"]["valid_images_file"],
        labels='inferred',
        label_mode='categorical',
        batch_size=config["train"]["batch_size"],
        image_size=(IMG_SIZE_val, IMG_SIZE_val))
    ## Augmentation de données : Sequence et Albumentations

    from albumentations import (Compose, Rotate, HorizontalFlip, VerticalFlip, Affine, RandomBrightnessContrast, ChannelShuffle, Crop)
    import albumentations as A

    # AUGMENTATIONS_TRAIN = Compose([
    #     Rotate(limit=[0,100], p=0.5),
    #     HorizontalFlip(p=0.5),
    #     VerticalFlip(p=0.5),
    #     Affine(shear=[-45, 45], p=0.5),
    #     RandomBrightnessContrast(p=0.5)
    # ])
    
    AUGMENTATIONS_TRAIN = Compose([
            Crop (x_min=0, y_min=0, x_max=224, y_max=224, always_apply=False, p=1.0)
        ])
    
    from tensorflow.keras.utils import Sequence
    import numpy as np
    import cv2 as cv

    class Aug_data(Sequence):
        # Initialisation de la séquence avec différents paramètres
        def __init__(self, x_train, y_train, batch_size, augmentations):
            self.x_train = x_train
            self.y_train = y_train
            self.classes = class_names
            self.batch_size = batch_size
            self.augment = augmentations
            self.indices1 = np.arange(len(x_train))
            np.random.shuffle(self.indices1) # Les indices permettent d'accéder
            # aux données et sont randomisés à chaque epoch pour varier la composition
            # des batches au cours de l'entraînement

        # Fonction calculant le nombre de pas de descente du gradient par epoch
        def __len__(self):
            return int(np.ceil(x_train.shape[0] / float(self.batch_size)))
        
        # Application de l'augmentation de données à chaque image du batch
        def apply_augmentation(self, bx, by):

            batch_x = np.zeros((bx.shape[0], IMG_SIZE_val, IMG_SIZE_val, 3))
            batch_y = by
            
            # Pour chaque image du batch
            for i in range(len(bx)):
                class_labels = []
                class_id = np.argmax(by[i])
                class_labels.append(self.classes[class_id])

                
                # Application de l'augmentation à l'image
                img = cv.imread(bx[i])
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (IMG_SIZE_train, IMG_SIZE_train)) #on resize l'image en 256 par 256
                img= self.augment(image=img)["image"] #on la crop aléatoirement en 224 par 224, ce qui fait office de data augmentation
                
                batch_x[i] = img
                
                
            return batch_x, batch_y

        # Fonction appelée à chaque nouveau batch : sélection et augmentation des données
        # idx = position du batch (idx = 5 => on prend le 5ème batch)
        def __getitem__(self, idx):
            batch_x = self.x_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
            batch_y = self.y_train[self.indices1[idx * self.batch_size:(idx + 1) * self.batch_size]]
            
            batch_x, batch_y = self.apply_augmentation(batch_x, batch_y)

            # Normalisation des données
            batch_x = tf.keras.applications.resnet.preprocess_input(batch_x)
            
            return batch_x, batch_y

        # Fonction appelée à la fin d'un epoch ; on randomise les indices d'accès aux données
        def on_epoch_end(self):
            np.random.shuffle(self.indices1)
    # Les images sont stockées avec les chemins d'accès
    import numpy as np

    x_train = np.array(train_ds.file_paths) #train_ds.file_paths renvoie la liste (type liste) des chemins d'accès aux images et np.array le passe en mode nd.array
    y_train = np.zeros((x_train.shape[0], nb_classes))

    ind_data = 0
    for bx, by in train_ds.as_numpy_iterator(): #as_numpy_iterator() permet de parcourir les données du dataset en les transformant en array numpy
        y_train[ind_data:ind_data+bx.shape[0]] = by
        ind_data += bx.shape[0]
    
    # Instanciation de la Sequence
    train_ds_aug = Aug_data(x_train, y_train, batch_size=32, augmentations=AUGMENTATIONS_TRAIN)# x_train est la liste des chemins d'accès aux images, y_train est la liste des labels sous forme de one-hot encoding

    # Normalisation des données de validation
    import numpy as np
    import tensorflow as tf

    x_val = np.zeros((31118, IMG_SIZE_val, IMG_SIZE_val, 3))#rentrer la taille du val
    y_val = np.zeros((31118, nb_classes))#rentrer la taille du val

    ind_data = 0
    for bx, by in validation_ds.as_numpy_iterator():
        
        x_val[ind_data:ind_data+bx.shape[0]] = bx
        y_val[ind_data:ind_data+bx.shape[0]] = by
        ind_data += bx.shape[0]
        
    x_val = tf.keras.applications.resnet.preprocess_input(x_val)
    
    
    ## Création du modèle
    from tensorflow.keras import regularizers
    from tensorflow.keras import optimizers
    import tensorflow as tf
    ### Poids d'imagenet
    conv_base = keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(IMG_SIZE_val, IMG_SIZE_val, 3),
        pooling=None,
        classes=nb_classes,
    )

    model = keras.Sequential(
        [  
            conv_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(nb_classes, kernel_regularizer=regularizers.L2(1e-4), activation='softmax')
        ]
    )
    model.summary()
    
    
    #Introduction de la focal Loss
    

    def categorical_focal_loss(gamma=2.0, alpha=0.25):
        """
        Implementation of Focal Loss from the paper in multiclass classification
        Formula:
            loss = -alpha*((1-p)^gamma)*log(p)
        Parameters:
            alpha -- the same as wighting factor in balanced cross entropy
            gamma -- focusing parameter for modulating factor (1-p)
        Default value:
            gamma -- 2.0 as mentioned in the paper
            alpha -- 0.25 as mentioned in the paper
        """
        def focal_loss(y_true, y_pred):
            # Define epsilon so that the backpropagation will not result in NaN
            # for 0 divisor case
            epsilon = K.epsilon()
            # Add the epsilon to prediction value
            #y_pred = y_pred + epsilon
            # Clip the prediction value
            y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
            # Calculate cross entropy
            cross_entropy = -y_true*K.log(y_pred)
            # Calculate weight that consists of  modulating factor and weighting factor
            weight = alpha * y_true * K.pow((1-y_pred), gamma)
            # Calculate focal loss
            loss = weight * cross_entropy
            # Sum the losses in mini_batch
            loss = K.sum(loss, axis=1)
            return loss
        
        return focal_loss
    
    if config["train"]["focal_loss"]:
        loss = categorical_focal_loss(gamma=5.0, alpha=1.0)
    else:
        loss = 'categorical_crossentropy'


    ## Entraînement du modèle
    # Ajout de l'optimiseur, de la fonction coût et des métriques
    lr = config["train"]["learning_rate"]
    model.compile(optimizers.SGD(learning_rate=lr, momentum=0.9), loss=loss, metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
  
    # Les callbacks
    #filepath = path to save the model at the end of each epoch

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=config["train"]["saved_weights_name"], 
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    #early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    #    monitor="val_categorical_accuracy",
    #    min_delta=0.01,
    #    patience=8,
    #    verbose=1,
    #    mode="auto")
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1,
                                patience=5, min_lr=0.00001, verbose=1)
    
    if config["train"]["train_sampling"]:
        #on définit le dictionnaire labels qui a chaque image de train associé son label
        path_to_train_images = config["data"]["train_images_file"]
        labels = {}
        for folder in list(os.listdir(path_to_train_images)):
            for images in list(os.listdir(os.path.join(path_to_train_images,folder))):
                labels[os.path.join(path_to_train_images,folder,images)] = folder
        
        #on définit list_IDs une liste qui contient tous les chemins des images du train
        list_IDs_train = []
        for folder in list(os.listdir(path_to_train_images)):
            for images in list(os.listdir(os.path.join(path_to_train_images,folder))):
                list_IDs_train.append(os.path.join(path_to_train_images,folder,images))
        
        history = model.fit(DataGenerator(list_IDs_train,labels, path_to_images=path_to_train_images, cap_sampling=150, config = config, n_classes=len(config["data"]["label"]), batch_size = config["train"]["batch_size"], crop = (224,224)), epochs=config["train"]["nb_epochs"], validation_data = (x_val, y_val), callbacks=[model_checkpoint_cb, reduce_lr_cb])
        
    elif config["train"]["val_sampling"]:
        path_to_val_images = config["data"]["valid_images_file"]
        labels = {}
        for folder in list(os.listdir(path_to_val_images)):
            for images in list(os.listdir(os.path.join(path_to_val_images,folder))):
                labels[os.path.join(path_to_val_images,folder,images)] = folder
        #on définit list_IDs une liste qui contient tous les chemins des images du train
        list_IDs_val = []
        for folder in list(os.listdir(path_to_val_images)):
            for images in list(os.listdir(os.path.join(path_to_val_images,folder))):
                list_IDs_val.append(os.path.join(path_to_val_images,folder,images))
        history = model.fit(train_ds_aug, epochs=config["train"]["nb_epochs"], validation_data = DataGenerator(list_IDs_val,labels, path_to_images=path_to_val_images, cap_sampling = 5, config = config, n_classes=len(config["data"]["label"]), batch_size = config["train"]["batch_size"], crop = (224,224)), callbacks=[model_checkpoint_cb, reduce_lr_cb])

    else:
        history = model.fit(train_ds_aug, epochs=config["train"]["nb_epochs"], validation_data = (x_val, y_val), callbacks=[model_checkpoint_cb, reduce_lr_cb]) 



    #on trace la categorical accuracy et la val_categorical_accuracy

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model categorical accuracy')
    plt.ylabel('categorical accuracy')
    plt.xlabel('epoch')


if __name__ == '__main__':
    _args = argparser.parse_args()
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    with tf.device('/GPU:' + gpu_id):
        _main_(_args)
    