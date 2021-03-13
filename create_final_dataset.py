import random
import math
import pandas as pd


file_with_paths = open("../images_in_bucket.txt")
lines = file_with_paths.readlines()

frames_per_location = {}
previous_location = ""
frames_in_location = []
min_number_frames_per_location = 9999

for line in lines:

    line = line.replace("\n", "")
    current_location = line.split("/")[-3]

    if current_location != previous_location and previous_location != "":
        frames_per_location[previous_location] = frames_in_location
        # Shuffling frames...
        random.shuffle(frames_per_location[previous_location])
        print("%s: %d images" % (previous_location, len(frames_in_location)))
        min_number_frames_per_location = min(min_number_frames_per_location, len(frames_in_location))
        frames_in_location = []

    frames_in_location.append(line)
    previous_location = current_location

frames_per_location[previous_location] = frames_in_location
# Shuffling frames...
random.shuffle(frames_per_location[previous_location])
print("%s: %d images" % (previous_location, len(frames_in_location)))
min_number_frames_per_location = min(min_number_frames_per_location, len(frames_in_location))

#For unbalanced dataset
room_name = []
nr_training_images_unbalanced = []
nr_validation_images_unbalanced = []
nr_testing_images_unbalanced = []

image_type_column_unbalanced = []
image_url_column_unbalanced = []
image_label_column_unbalanced = []

#For balanced_dataset
nr_training_images_balanced = []
nr_validation_images_balanced = []
nr_testing_images_balanced = []

image_type_column_balanced = []
image_url_column_balanced = []
image_label_column_balanced = []

for key, value in frames_per_location.items():

    split_label = key.split("_")
    final_label = ""
    for word in split_label:
        final_label += word.upper() + " "

    print("Treating room: %s" %final_label)
    room_name.append(final_label)

    number_of_training_frames = int(math.ceil(len(frames_per_location[key]) * 0.80))
    number_of_validation_frames = int(math.ceil(len(frames_per_location[key]) * 0.10))
    number_of_testing_frames = int(math.ceil(len(frames_per_location[key]) * 0.10))

    #SLICING - First: Inclusive | Second: Exclusive

    training_frames_unbalanced = frames_per_location[key][0:number_of_training_frames]
    nr_training_images_unbalanced.append(len(training_frames_unbalanced))
    for frame in training_frames_unbalanced:
        image_type_column_unbalanced.append("TRAIN")
        image_url_column_unbalanced.append(frame)
        image_label_column_unbalanced.append(final_label)

    training_frames_balanced = training_frames_unbalanced[0:int(min_number_frames_per_location * 0.80)]
    nr_training_images_balanced.append(len(training_frames_balanced))
    for frame in training_frames_balanced:
        image_type_column_balanced.append("TRAIN")
        image_url_column_balanced.append(frame)
        image_label_column_balanced.append(final_label)

    last_index_validation_frames = (number_of_training_frames + number_of_validation_frames)
    validation_frames_unbalanced = frames_per_location[key][number_of_training_frames:last_index_validation_frames]
    nr_validation_images_unbalanced.append(len(validation_frames_unbalanced))
    for frame in validation_frames_unbalanced:
        image_type_column_unbalanced.append("VALIDATION")
        image_url_column_unbalanced.append(frame)
        image_label_column_unbalanced.append(final_label)

    validation_frames_balanced = validation_frames_unbalanced[0:int(min_number_frames_per_location * 0.10)]
    nr_validation_images_balanced.append(len(validation_frames_balanced))
    for frame in validation_frames_balanced:
        image_type_column_balanced.append("VALIDATION")
        image_url_column_balanced.append(frame)
        image_label_column_balanced.append(final_label)

    last_index_testing_frames = len(frames_per_location[key])
    testing_frames_unbalanced = frames_per_location[key][last_index_validation_frames:last_index_testing_frames]
    nr_testing_images_unbalanced.append(len(testing_frames_unbalanced))
    for frame in testing_frames_unbalanced:
        image_type_column_unbalanced.append("TEST")
        image_url_column_unbalanced.append(frame)
        image_label_column_unbalanced.append(final_label)

    testing_frames_balanced = testing_frames_unbalanced[0:int(min_number_frames_per_location * 0.10)]
    nr_testing_images_balanced.append(len(testing_frames_balanced))
    for frame in testing_frames_balanced:
        image_type_column_balanced.append("TEST")
        image_url_column_balanced.append(frame)
        image_label_column_balanced.append(final_label)

dataframe = {"Type": pd.Series(image_type_column_unbalanced), "URL": pd.Series(image_url_column_unbalanced), "Label": pd.Series(image_label_column_unbalanced)}
dataset = pd.DataFrame(dataframe)
dataset.to_csv("/Users/tomasmamede/Desktop/dataset_unbalanced.csv", header=False, index=False)

dataframe = {"Room Name": pd.Series(room_name), "Training Images": pd.Series(nr_training_images_unbalanced), "Validation Images": pd.Series(nr_validation_images_unbalanced), "Testing Images": pd.Series(nr_testing_images_unbalanced)}
summary = pd.DataFrame(dataframe)
summary.to_csv("/Users/tomasmamede/Desktop/count_dataset_unbalanced.csv", index=False)

dataframe = {"Type": pd.Series(image_type_column_balanced), "URL": pd.Series(image_url_column_balanced), "Label": pd.Series(image_label_column_balanced)}
dataset = pd.DataFrame(dataframe)
dataset.to_csv("/Users/tomasmamede/Desktop/dataset_balanced.csv", header=False, index=False)

dataframe = {"Room Name": pd.Series(room_name), "Training Images": pd.Series(nr_training_images_balanced), "Validation Images": pd.Series(nr_validation_images_balanced), "Testing Images": pd.Series(nr_testing_images_balanced)}
summary = pd.DataFrame(dataframe)
summary.to_csv("/Users/tomasmamede/Desktop/count_dataset_balanced.csv", index=False)

