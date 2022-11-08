#importations

import json
import os 
import cv2
#path

config_path = "/home/acarlier/code/code_to_present_class_imbalanced_issue/json_data/caltech_images_20210113.json"
path_to_cct = "/home/acarlier/code/code_to_present_class_imbalanced_issue/caltech_trap"

#link to json

with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
#on crée le dictionnaire des annotations

annot = {}
for dic in config["categories"]:
    annot[dic["id"]] = dic["name"]
print(annot)
      
#on crée des folder où on range les images de chaque classe

try:
    os.mkdir("/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_images_trap/")
except:
    pass

for dic in config["annotations"]:
    try:
        os.mkdir("/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_images_trap/" + annot[dic["category_id"]])
    except:
        pass
    print(dic["image_id"])
    img = cv2.imread(os.path.join(path_to_cct, dic["image_id"] + ".jpg"))
    cv2.imwrite("/home/acarlier/code/code_to_present_class_imbalanced_issue/Caltech_images_trap/" + annot[dic["category_id"]] + "/" + dic["image_id"] + ".jpg", img)
    