# Téléchargement de la base de données
# Pour récupérer le nombre de classes du training dataset
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display
import json
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
import tensorflow
from tensorflow.keras.metrics import Accuracy
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from PIL import Image

config_path = '/home/acarlier/code/code_to_present_class_imbalanced_issue/config/plantnet_val_sampling.json'

with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

IMG_SIZE = 224
train_ds = keras.utils.image_dataset_from_directory(
    directory=config["data"]["train_images_file"],
    labels='inferred',
    label_mode='categorical',
    batch_size=config["train"]["batch_size"],
    image_size=(IMG_SIZE, IMG_SIZE))
class_names = train_ds.class_names
print(class_names)
nb_classes = len(class_names)
print(nb_classes)
# Chargement du modèle


# Création de l'architecture du modèle à utiser

conv_base = keras.applications.resnet.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
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

# Chargement des poids
model.load_weights(config["train"]["saved_weights_name"])#mettre ici le même chemin que celui des callbacks (dans la var model_checkpoints)

model.compile(optimizers.SGD(learning_rate=1e-3, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
# Chargement de l'ensemble de test
# Récupération du dataset de test
test_ds = keras.utils.image_dataset_from_directory(
    directory=config["data"]["test_images_file"],
    labels='inferred',
    label_mode='categorical',
    batch_size=config["train"]["batch_size"],
    image_size=(IMG_SIZE, IMG_SIZE))
test_ds_no_shuffle = keras.utils.image_dataset_from_directory(
    directory=config["data"]["test_images_file"],
    labels='inferred',
    label_mode='categorical',
    shuffle=False,
    batch_size=config["train"]["batch_size"],
    image_size=(IMG_SIZE, IMG_SIZE))
import numpy as np

nb_images = 31112
x_test = np.zeros((nb_images, IMG_SIZE, IMG_SIZE, 3))
y_test = np.zeros((nb_images, nb_classes))

ind_data = 0
for bx, by in test_ds_no_shuffle.as_numpy_iterator():
  
  x_test[ind_data:ind_data+bx.shape[0]] = bx
  y_test[ind_data:ind_data+bx.shape[0]] = by

  ind_data += bx.shape[0]

x_test = tf.keras.applications.resnet.preprocess_input(x_test)


plt.imshow(x_test[0].astype(int))
print(test_ds_no_shuffle.file_paths[0])


# Analyse des résultats : précision, sensibilité, F1_score
y_pred = model.predict(x_test)

test_accuracy = Accuracy()

prediction = tensorflow.argmax(y_pred, axis=1, output_type=tensorflow.int32)

true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(y_pred, axis=1)


#calcul de l'accuracy par classe 

nb_images_par_classe = {label : 0 for label in class_names}
for i in range(nb_classes):
    nb_images_par_classe[class_names[i]] = np.sum(true_labels == i)

accuracy = {label : 0 for label in class_names}
for i in range(nb_classes):
    accuracy[class_names[i]] = np.sum((true_labels == i) & (predicted_labels == i))/nb_images_par_classe[class_names[i]]

    
print(accuracy)
        
#on calcule la macro average top-1 accuracy, ou moyenne arithmétique (micro pour la weighted)

somme = 0
for classe in accuracy:
    somme += accuracy[classe]
    
print("La macro average top-1 accuracy est de:", somme/nb_classes) #macro average = moyenne arithmétique 
        
#On alcule la top-1 accuracy (calucle de l'accuracy indépendemment des classes), ou accuracy weighted

somme = 0
for i in range(len(true_labels)):
    if predicted_labels[i] == true_labels[i]:
            somme += 1
             
print("La top-1 accuracy est de:", somme/nb_images) #top1 accuracy = accuracy weighted = moyenne pondérée

#on veut maintenant calculer la moyenne des accuracy pour des types de classes variant selon leur nombre d'images



exit()
report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=5, output_dict=True)

print(report)