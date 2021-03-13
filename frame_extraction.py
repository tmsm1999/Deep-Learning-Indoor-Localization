import cv2
import shutil
import math
import os
import pandas as pd
import numpy as np


def frame_extraction(frames_per_room, is_to_delete_frames):

    cur_dir = os.getcwd()
    os.chdir("../Dataset Museu/Categorias")

    #Get room names
    rooms_directory = os.getcwd()
    rooms = []  # Museum rooms
    rooms_names = []
    frames_count = []
    for dir in os.listdir(rooms_directory):

        if dir == ".DS_Store": continue
        rooms.append(os.path.join(rooms_directory, dir))

    print("%d categories in the dataset" % len(rooms))

    for room in rooms:
        print("\n")

        #Change to a new room directory.
        os.chdir(room)

        room_name_parts = room.split("/")
        room_name = room_name_parts[len(room_name_parts) - 1]
        print("Analysing room:  %s ..." % room_name)
        rooms_names.append(room_name)

        for subdirs, dirs, files in os.walk(os.getcwd()):

            # Remove everything before getting frames from videos.
            if os.path.isdir("Frames"):
                shutil.rmtree("Frames")
                print("Frames directory was removed ...")

            # NOTE: Uncomment this line to remove all folders.
            if is_to_delete_frames:
                continue

            # Create new directory to save frames of the video.
            new_dir_path = os.path.join(os.getcwd(), "Frames")
            os.mkdir(new_dir_path)
            print("Frames directory was created ...")

            #In macOS we need to ignore .DS_Store
            number_of_files_in_folder = len(files)
            if ".DS_Store" in files:
                print(".DS_Store is present.")
                number_of_files_in_folder -= 1

            print("Total number of files in folder: %d" % number_of_files_in_folder)

            number_of_frames_per_video = frames_per_room / number_of_files_in_folder
            print("Total number of frames per video: %d" % number_of_frames_per_video)

            count = 0

            #Go through all videos of the room.
            for file in files:

                if file == ".DS_Store":
                    print("Ignoring .DS_Store ...")
                    continue

                video_name = file.split(".")[0]

                print("Getting frames from video %s ..." % video_name)
                video_absolute_path = os.path.abspath(file)
                video_object = cv2.VideoCapture(video_absolute_path)

                total_frame_count = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Total number of frames is: %d" % total_frame_count)

                step = int(math.ceil(total_frame_count / number_of_frames_per_video))

                if step < 1 or (number_of_files_in_folder == 1 and total_frame_count - 200 < frames_per_room):
                    step = 1
                print("Step: %d" % step)

                current_frame = 0

                has_next_frame = True
                while has_next_frame:

                    has_next_frame, frame = video_object.read()
                    if current_frame % step == 0:

                        if type(frame) is np.ndarray:

                            count += 1
                            cv2.imwrite(os.path.join(new_dir_path, room_name + "_frame_" + str(count) + ".jpeg"), frame)

                    current_frame += 1

            frames_count.append(count)

    if is_to_delete_frames:
        exit(0)

    #Dataframe com contagem dos frames por sala

    rooms_names_series = pd.Series(rooms_names)
    frames_count_series = pd.Series(frames_count)

    dataframe = {"Room Name": rooms_names_series, "Contagem": frames_count_series}
    result = pd.DataFrame(dataframe)
    result.sort_values(by="Contagem", inplace=True, ascending=False)
    result.to_csv("/Users/tomasmamede/Desktop/Contagem.csv", index=False)

#First argument passes to the function the number of frames we want per room.
#Second argument passes to the function if we wish to remove all images from datatset
frame_extraction(500, 0)
