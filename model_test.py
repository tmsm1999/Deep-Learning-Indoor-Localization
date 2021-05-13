import cv2
import model_inference
import getopt
import sys

# For moving average
queue = []
window_size = 0

try:
    opts, args = getopt.getopt(sys.argv, "ho:v", ["help", "output="])
    window_size = int(args[1])
except getopt.GetoptError as err:
    print("Wrong input arguments. Try: script_name window_size")
    sys.exit(1)

model = model_inference.Model("model.tflite", "dict.txt")

video_file_path = "/Users/tomasmamede/Library/Mobile Documents/com~apple~CloudDocs/Drive/Faculdade/Investigação/Projeto Safe Cities/Dataset Museu/Vídeos de Teste Museu/IMG_2353.MOV"
capture = cv2.VideoCapture(video_file_path)

count = 0

while capture.isOpened():

    ret, frame = capture.read()

    if not ret:
        break

    # Run inference on the model
    results_raw_frame = model.classify(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None)

    if len(results_raw_frame) > 0:

        if len(queue) < window_size:
            queue.append(results_raw_frame[0])
        else:

            queue.pop()
            queue.insert(0, results_raw_frame[0])
            print(queue)

            confidence = {}
            total = 0
            for res in queue:
                if res[0] in confidence:
                    confidence[res[0]] += res[1]
                else:
                    confidence[res[0]] = res[1]

            max_confidence = -1.0
            max_room = ""
            for room in confidence:
                result = confidence[room] / window_size
                if result > max_confidence:
                    max_confidence = result
                    max_room = room

            text_to_show = ""
            if max_confidence > 0.25:
                text_to_show = max_room + ", " + "{:.2f}%".format(max_confidence * 100)
            else:
                text_to_show = "Below confidence threshold"

            font = cv2.FONT_HERSHEY_PLAIN
            org = (20, 100)
            fontScale = 3
            color = (255, 255, 255)
            thickness = 4
            line_type = cv2.LINE_8
            final_frame = cv2.putText(frame, str(text_to_show), org, font, fontScale, color, thickness, line_type)

    count += 1
    if count % 10:
        continue

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
