# Téléchargement de la base de données
# Pour récupérer le nombre de classes du training dataset
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display
import json

config_path = '/home/acarlier/code/code_to_present_class_imbalanced_issue/config/config_trap_classique.json'

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
# from google.colab import drive
# drive.mount('/content/drive')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
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

x_test = np.zeros((11757, IMG_SIZE, IMG_SIZE, 3))
y_test = np.zeros((11757, nb_classes))

ind_data = 0
for bx, by in test_ds_no_shuffle.as_numpy_iterator():
  
  x_test[ind_data:ind_data+bx.shape[0]] = bx
  y_test[ind_data:ind_data+bx.shape[0]] = by

  ind_data += bx.shape[0]

x_test = tf.keras.applications.resnet.preprocess_input(x_test)
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from PIL import Image

plt.imshow(x_test[0].astype(int))
print(test_ds_no_shuffle.file_paths[0])
# Analyse des résultats : précision, sensibilité, F1_score
y_pred = model.predict(x_test)
import tensorflow
from tensorflow.keras.metrics import Accuracy
test_accuracy = Accuracy()

prediction = tensorflow.argmax(y_pred, axis=1, output_type=tensorflow.int32)
print(prediction)
test_accuracy(prediction, np.argmax(y_test, axis=1))

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
test_topk_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_k_categorical_accuracy", dtype=None)

test_topk_accuracy(y_test, y_pred)
print("Test set top 2 accuracy: {:.3%}".format(test_topk_accuracy.result()))
## Précision et Sensibilité
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(y_pred, axis=1)

#report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=5)
report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=5, output_dict=True)
print(report)
import pandas as pd
df = pd.DataFrame(report).transpose()
display(df)

# Proportion de classes dont le f1_score est inférieur au f1_score moyen parmi 
# les classes dont le support est inférieur à un seui

df_without_average = df.head(nb_classes)
averages = df.tail(3)
display(averages)

L = []
for folder in report:
    if folder != 'accuracy' and folder != 'macro avg' and folder != 'weighted avg':
        L.append(report[folder]['f1-score'])
        L.append(list(report[folder].values())[2])
print("L'écart type du F1-score entre les classes est:",np.std(L))

# seuil = 15
# selection = df_without_average.loc[df_without_average['support'] <= seuil]
# display(selection)

# nb_classes_under_represented = len(selection)

# bad_f1_score = selection.loc[selection['f1-score'] <= averages.at['macro avg', 'f1-score']]
# display(bad_f1_score)

# nb_classes_bad_f1_score = len(bad_f1_score)
# proportion = (nb_classes_bad_f1_score / nb_classes_under_represented) * 100
# print("------- Proportion de : " + str(proportion) + " % -------")
