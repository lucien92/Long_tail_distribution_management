import json
import os
import statistics

#path to data

config_path = "/home/acarlier/code/code_to_present_class_imbalanced_issue/stat_on_data_imbalanced_issue/config.json"

#analyse des données

with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())
    
#we count the number of images for each species

count_species = config["caltech_trap"]
print(count_species)

#Première mesure du déséquilibre de classe: le nombre de classe dont le rapport de leur nombre d'image sur le nombre d'images de la classe majoritaire est supérieur à 10%

class_with_less_images = min(count_species, key=count_species.get)
class_with_more_images = max(count_species, key=count_species.get)

print("Le rapport entre le nombre d'images dans la classe avec le moins d'images et le nombre d'images dans la classe avec le plus d'images est de: ", count_species[class_with_less_images]/count_species[class_with_more_images])
print( f"C'est-dire:{count_species[class_with_less_images]/count_species[class_with_more_images]*100}%", )

count = 0
for key, value in count_species.items():
    if value/count_species[class_with_more_images] < 0.1:
        count += 1
print(f"Le nombre de classe dont le rapport du nombre d'images sur le nombre d'images de la classe majoritaire est inférieur à 10% est de: {count} sur {len(count_species)}")
print("Ainsi le pourcentage de classe dont le nombre d'images est inférieur à 10% du nombre d'images de la classe majoritaire est de: ", count/len(count_species)*100, "%")

if count>20:
    print("Le problème est donc bien présent, surtout si on accord de l'importance aux espèces rare comme dans l'étude du comportement animal")

#Deuxième mesure: on introduit une  mesure qui nous renseigne sur la distribution des images dans les classes, l'écart-type

print("L'écart-type des classes est de: ", statistics.stdev(count_species.values()))