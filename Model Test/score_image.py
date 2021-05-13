import warnings
# Supress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import tfmodel

model = tfmodel.Model("model.tflite", "dict.txt")
frames_directory = "/Users/tomasmamede/Documents/Investigação/Dataset Museu/Teste/Atrio/Frames"

total = 0
correct = 0

for image in os.listdir(frames_directory):

    if image == ".DS_Store":
        continue

    path_to_image = os.path.join(frames_directory, image)
    res_list = model.classify(path_to_image, max_results=5, min_confidence=0.01)
    print('{},{},{:.2f}'.format(image.split(".")[0], res_list[0][0], res_list[0][1]))

    if image.split("_")[0] == res_list[0][0]:
        correct += 1

    total += 1

print("%d correct predictions over %d input images." % (correct, total))