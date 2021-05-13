import cv2
import model_inference
import tfmodel
import math
import os

video_file_path = "/Users/tomasmamede/Documents/Investigação/Dataset Museu/Teste/Salas/IMG_2353.MOV"
capture = cv2.VideoCapture(video_file_path)

total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(math.ceil(total_frame_count / 500))

frames_directory = "/Users/tomasmamede/Desktop/Model_Validation_Frames_Salas/"
frames_to_classify = os.listdir(frames_directory)
frames_to_classify.sort()
#print(frames_to_classify)
#print(len(frames_to_classify))

total_frames = 0
frame_counter = 0
count = 0

model_jpeg = tfmodel.Model("model.tflite", "dict.txt")
model_raw = model_inference.Model("model.tflite", "dict.txt")

print("IMAGE_NAME,RES_LABEL_RAW_FRAME,CONF_LABEL_RAW_FRAME,RES_LABEL_JPEG_IMAGE,CONF_LABEL_JPEG_IMAGE")

correct_raw_frames = 0
correct_jpeg_images = 0

while capture.isOpened():

    ret, frame = capture.read()
    frame_counter += 1

    if not ret:
        break

    if frame_counter % step != 0:
        continue

    count += 1
    #cv2.imwrite(frames_directory + "/x_frame_%d.jpeg" % count, frame)
    #continue

    for frame_in_folder in frames_to_classify:
        if "_" + str(count) + "." in frame_in_folder:

            # print("Found! %s" % frame_in_folder)
            # continue

            # Image saved to disk.
            path_jpeg_image = "/Users/tomasmamede/Desktop/Model_Validation_Frames_Salas/" + frame_in_folder
            cv2.imwrite(path_jpeg_image, frame)

            res_label_raw_frame = " "
            res_conf_raw_frame = 0.0
            res_label_jpeg_img = " "
            res_conf_jpeg_img = 0.0

            # Alteração da matriz RGB.
            results_raw_frame = model_raw.classify(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 5, 0.01)
            results_jpeg_image = model_jpeg.classify(path_jpeg_image, 5, 0.01)

            if len(results_raw_frame) >= 1:
                #print(results_raw_frame)
                res_label_raw_frame = results_raw_frame[0][0]
                res_conf_raw_frame = results_raw_frame[0][1]

                if frame_in_folder.split("_")[0] == res_label_raw_frame:
                    correct_raw_frames += 1

            if len(results_jpeg_image) >= 1:
                #print(res_conf_jpeg_img)
                res_label_jpeg_img = results_jpeg_image[0][0]
                res_conf_jpeg_img = results_jpeg_image[0][1]

                if frame_in_folder.split("_")[0] == res_label_jpeg_img:
                    correct_jpeg_images += 1

            print('{},{},{:.2f},{},{:.2f}'.format(frame_in_folder.split(".")[0], res_label_raw_frame, res_conf_raw_frame,
                                                  res_label_jpeg_img, res_conf_jpeg_img))
            total_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

capture.release()
cv2.destroyAllWindows()

print("Correct raw frames predictions: %d" % correct_raw_frames)
print("Correct jpeg images predictions: %d" % correct_jpeg_images)
