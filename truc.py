import os
L = []
for folder in os.listdir("/home/acarlier/code/code_to_present_class_imbalanced_issue/plantnet_300K/images_train"):
    L.append(folder)
    
A = []
for folder in os.listdir("/home/acarlier/code/code_to_present_class_imbalanced_issue/plantnet_300K/images_val"):
    A.append(folder)
    
print(A == L)
print(A)