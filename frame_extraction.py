import cv2
import shutil
import os
import numpy as np


def frame_extraction(frames_per_room):

    cur_dir = os.getcwd()
    os.chdir("../Dataset Museu/Categorias")

    #Get room names
    rooms_directory = os.getcwd()
    rooms = []  # Museum rooms
    for dir in os.listdir(rooms_directory):

        if dir == ".DS_Store": continue
        rooms.append(os.path.join(rooms_directory, dir))

    print("%d rooms in the dataset" % len(rooms))

    for room in rooms:
        print("\n")

        #Change to a new room directory.
        os.chdir(room)
        room_name_parts = room.split("/")
        print("Analysing room:  %s ..." % room_name_parts[len(room_name_parts) - 1])
        for subdirs, dirs, files in os.walk(os.getcwd()):

            number_of_files_in_folder = len(files)
            number_of_frames_per_video = frames_per_room / number_of_files_in_folder

            #Go through all videos of the room.
            for file in files:

                if file == ".DS_Store":
                    continue

                video_name = file.split(".")[0]
                video_frames_folder_name = "%s Frames" % video_name
                #Remove everything before getting frames from videos.
                if os.path.isdir(video_frames_folder_name):
                    shutil.rmtree(video_frames_folder_name)
                    print("%s was removed ..." % video_frames_folder_name)

                # NOTE: Uncomment this line to remove all folders.
                continue

                #Create new directory to save frames of the video.
                new_dir_path = os.path.join(os.getcwd(), video_frames_folder_name)
                os.mkdir(new_dir_path)
                print("%s directory was created ..." % video_frames_folder_name)

                print("Getting frames from video %s ..." % video_name)
                video_absolute_path = os.path.abspath(file)
                video_object = cv2.VideoCapture(video_absolute_path)

                total_frame_count = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Total number of frames is: %d" % total_frame_count)

                step = int(total_frame_count / number_of_frames_per_video)
                current_frame = 0
                count = 1

                has_next_frame = True
                while has_next_frame:

                    has_next_frame, frame = video_object.read()
                    if current_frame % step == 0:

                        if type(frame) is np.ndarray:

                            cv2.imwrite(os.path.join(new_dir_path, "frame_" + str(count) + ".jpeg"), frame)
                            count += 1

                    current_frame += 1


#frames_per_video = int(input("Number of frames per video: "))
frame_extraction(150)
