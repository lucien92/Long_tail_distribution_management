from imghdr import tests
import os

#paths 
train = "/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech101_images_train_val_test/caltech_images_train"
val = "/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech101_images_train_val_test/caltech_images_val"
test = "/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech101_images_train_val_test/caltech_images_test"

#vérifications

#train/val

L = []
for folder in os.listdir(train):
    for images in os.listdir(os.path.join(train, folder)):
        L.append(os.path.join(folder,images)) #on rajouter le folder car les images son tnumérotées de 1 à N dans chaque dossier 
print(L)
c = 0       
for folder in os.listdir(val):
    for images in os.listdir(os.path.join(val, folder)):
        if images in L:
            c +=1
print(c)
if c == 0:
    print("Il n'y a pas d'images en commun entre test et val")   
               
#train/test

L = []
for folder in os.listdir(train):
    for images in os.listdir(os.path.join(train, folder)):
        L.append(os.path.join(folder,images))
b = 0       
for folder in os.listdir(test):
    for images in os.listdir(os.path.join(test, folder)):
        if images in L:
            #print("Il y a des images en commun entre train et test")
            b +=1
print(b)
if b == 0:
    print("Il n'y a pas d'images en commun entre test et val")   
    
#test et val

L = []
for folder in os.listdir(test):
    for images in os.listdir(os.path.join(test, folder)):
        L.append(os.path.join(folder,images))
a = 0        
for folder in os.listdir(test):
    for images in os.listdir(os.path.join(test, folder)):
        if images in L:
            #print("Il y a des images en commun entre train et test")
            a += 1
print(a)
if a == 0:
    print("Il n'y a pas d'images en commun entre test et val")         

