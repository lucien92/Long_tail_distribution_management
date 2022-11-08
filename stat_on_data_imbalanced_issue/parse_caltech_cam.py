import json
config_path = '/home/acarlier/code/code_to_present_class_imbalanced_issue/config_json/SnapshotKgalagai_S1_v1.0.json'

with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())



# categories = {
#   "id" : int,
#   "name" : str #donne le nom associé à l'id
# }

# annotation = {
#   "id" : str,
#   "image_id" : str, #donne id de l'image
#   "category_id" : int #donne l'id de l'animal sur l'image
# }

#on veut compter le nombre d'images par id

ref = {}
for i in config['categories']:
    ids = i['id']
    specie = i['name']
    ref[ids] = specie

#on veut compter le nombre d'images par id

count = {espece:0 for espece in ref.values()}
for i in config['annotations']:
    specie_id = i['category_id']
    specie = ref[specie_id]
    count[specie] += 1
    print(count)
    
print(count)
