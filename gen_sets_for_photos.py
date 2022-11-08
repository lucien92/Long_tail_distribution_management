#importations

import os
import pandas as pd
import cv2

#paths

path_to_data = '/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_DB'
train_csv = '/home/acarlier/code/code_to_present_class_imbalanced_issue/csv_caltech_101/train_caltech_trap.csv'
validation_csv = '/home/acarlier/code/code_to_present_class_imbalanced_issue/csv_caltech_101/val_caltech_trap.csv'
test_csv = '/home/acarlier/code/code_to_present_class_imbalanced_issue/csv_caltech_101/test_caltech_trap.csv'

#dataframe

df_train = pd.read_csv(train_csv)
print(df_train['images'])
df_val = pd.read_csv(validation_csv)
print(df_val)
df_test = pd.read_csv(test_csv)
print(df_test)


#on crée les répertoires
try:
    os.mkdir("/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test")
except:
    pass
try:
    os.mkdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_train')
    os.mkdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_val')
    os.mkdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_test')
except:
    pass
#on veut afficher la première ligne du dataframe

print(df_train. iloc[0,0])
print(df_val)


#on veut générer les même ensembles train, val et test qui se trouve dans le dossier csv_caltech_101 mais avec les images correspondantes de Caltech-101-DBCaltech101

for folder in os.listdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_DB'):
    try:
        os.mkdir(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_train/{folder}')
    except:
        pass
    
    path_to_folder = os.path.join(path_to_data,folder)
    for images in os.listdir(os.path.join(folder,path_to_folder)):
        path_to_images = os.path.join(path_to_folder,images)
        print(path_to_images)
        if path_to_images in df_train['images'].to_dict().values(): #in df train: /home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech-101-DBCaltech101/sunflower/image_0012.jpg
            print('train')
            try:
                img = cv2.imread(path_to_images)
                cv2.imwrite(str(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_train/{folder}/{images}'),img)
            except:
                pass


for folder in os.listdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_DB'):
    try:
        os.mkdir(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_val/{folder}')
    except:
        pass
    path_to_folder = os.path.join(path_to_data,folder)
    for images in os.listdir(os.path.join(folder,path_to_folder)):
        path_to_images = os.path.join(path_to_folder,images)
        if path_to_images in df_val['images'].to_dict().values():
                print("validation")
                try:
                    img = cv2.imread(path_to_images)
                    cv2.imwrite(str(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_val/{folder}/{images}'),img)
                except:
                    pass   

for folder in os.listdir('/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_DB'):

    try:
        os.mkdir(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_trap_images_train_val_test/caltech_images_test/{folder}')
    except:
        pass
    path_to_folder = os.path.join(path_to_data,folder)
    for images in os.listdir(os.path.join(folder,path_to_folder)):
        path_to_images = os.path.join(path_to_folder,images)
        if path_to_images in df_test['images'].to_dict().values():
                print("test")
                try:
                    img = cv2.imread(path_to_images)
                    cv2.imwrite(str(f'/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_images_train_val_test/caltech_images_test/{folder}/{images}'),img)
                except:
                    pass