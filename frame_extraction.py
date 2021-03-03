import cv2
import shutil
import os
import numpy as np


def frame_extraction():

    cur_dir = os.getcwd()
    os.chdir("../Dataset Museu - Iteração 1")

    #Get room names
    rooms_directory = os.getcwd()
    rooms = []  # Museum rooms
    for dir in os.listdir(rooms_directory):

        if dir == ".DS_Store": continue
        rooms.append(os.path.join(rooms_directory, dir))

    print("%d rooms in the dataset" % len(rooms))

    for room in rooms:

        #Change to a new room directory.
        os.chdir(room)
        room_name_parts = room.split("/")
        print("Analysing room:  %s ..." % room_name_parts[len(room_name_parts) - 1])
        for subdirs, dirs, files in os.walk(os.getcwd()):

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
                # continue

                #Create new directory to save frames of the video.
                new_dir_path = os.path.join(os.getcwd(), video_frames_folder_name)
                os.mkdir(new_dir_path)
                print("%s directory was created ..." % video_frames_folder_name)

                print("Getting frames from video %s ..." % video_name)
                video_absolute_path = os.path.abspath(file)
                video_object = cv2.VideoCapture(video_absolute_path)

                all_frames = []
                has_next_frame = True

                while has_next_frame:

                    has_next_frame, frame = video_object.read()
                    all_frames.append(frame)

                total_number_of_frames = len(all_frames)
                print("Total number of frames: %d" % total_number_of_frames)

                count = 1
                step = int(total_number_of_frames / 150)
                for i in range(1, total_number_of_frames, step):

                    if (type(all_frames[i]) is np.ndarray):

                        cv2.imwrite(os.path.join(new_dir_path, "frame_" + str(count) + ".jpeg"), all_frames[i])
                        count += 1

frame_extraction()
