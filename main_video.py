import csv
import json
import os
from pathlib import Path

import cv2

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

save_dir=Path('/home/federico/Data/Human_motion_videos/Object_Detection/')

# Open the video file
videos_path = Path("/home/federico/Data/Human_motion_videos/output_videos/")
video_path = Path("/home/federico/Videos/video.mp4")
# cap = cv2.VideoCapture(video_path)
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True, save_txt=True, save_conf=True)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
list_of_files=[]
for person_folder in videos_path.iterdir():
    for action_folder in person_folder.iterdir():
        for trial_fodler in action_folder.iterdir():
            # save_dir_full = save_dir / person_folder.name / action_folder.name / trial_fodler.name
            # save_dir_full.mkdir(exist_ok=True, parents=True)
            print(trial_fodler/'video.mp4')
            aa = f'{person_folder.name}_{action_folder.name}_{trial_fodler.name}'
            print(aa)
            list_of_files.append(str(trial_fodler/'video.mp4'))
            results = model.track(trial_fodler/'video.mp4', persist=True, save_txt=False, conf=0.25)# stream=True

            # results.save_txt(txt_file=f'/home/federico/Downloads/Fede/{aa}',save_conf=True)
            for frame, r in enumerate(results):
                r.save_txt(txt_file=f'/home/federico/Downloads/Fede/{aa}.txt',save_conf=True)
                with open(f'/home/federico/Downloads/Fede/{aa}.txt','a') as f:
                    f.writelines(f'{frame}\n')

                # withlump(r.tojson(decimals=3), file)
            # print('1')
# print(results)


with open(str(save_dir/'list_videos.txt'), 'w', newline='') as file:
    # Step 4: Using csv.writer to write the list to the CSV file
    writer = csv.writer(file)
    writer.writerow(list_of_files) # Use writerow for single list