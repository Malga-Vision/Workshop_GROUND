import json
from collections import namedtuple
from pathlib import Path

import cv2
import jsonpickle
import matplotlib
import numpy as np
import pandas as pd
import torch

import load_rgb
from bboxes_load import load_bboxes, xywh_to_xyxy, xywhn2xyxy, float2pixel, load_frames_with_bboxes
from hpe_load import load_hpe, draw_axis_3d, estimate_visual_cone_on_table
from kpts_data_base_load import extract_one_joint, extract_one_action_sample
from load_kpts_yolo import load_frames_with_bboxes_kpts


def dist_points(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


if __name__ == '__main__':
    # actions = ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
    #            'touch_rubiks_cube',
    #            'transport_bottle', 'transport_pen', 'transport_rubiks_cube']

    person = 14
    trial=1
    action = 'transport_bottle'
    target_obj_string = 'bottle'
    target_obj_string_to_write = 'bottle'
    follow=False
    plot_on_screen=False
    target_is_point=True
    Point= namedtuple('Point', ['x1', 'y1', 'fake'])
    target_point=Point(0.907543, 0.749613, False)

    # kpts_root_folder_data = Path('/home/federico/Data/Human_Motion')  # positions_3d_centered_shortened.csv
    #
    # my_dataset = pd.read_csv(kpts_root_folder_data / 'positions_3d_reformat.csv', header=[0])
    # col = my_dataset.columns.values.tolist()
    # my_data = extract_one_joint(my_dataset, 'rwrist')
    # kpts = extract_one_action_sample(person=14, action='transport_bottle', trial=0)

    head_pose = load_hpe(person=person, action=action, trial=trial)

    frames_bbox = load_frames_with_bboxes(person=person, action=action, trial=trial)

    # frame_files = load_rgb.load_rgb_name(person=14, action='transport_bottle', trial=0)
    i =0
    video_file = load_rgb.load_rgb_video_path(person=person, action=action, trial=trial)

    kpts_yolo = load_frames_with_bboxes_kpts(person=person, action=action, trial=trial)
    i_max = min(len(head_pose), len(kpts_yolo), len(frames_bbox))
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    window_name= ' Federico'
    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    # SET VARIABLE FOR SAVING
    dist_hand_target =[]
    dist_head_target =[]
    detected =False
    target =0,0

    intial_position =None


    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
    # Calculate the delay between frames
    frame_delay = int(1000 / fps)  # delay in milliseconds

    while cap.isOpened() and i<i_max:
        ret, image = cap.read()
        if not ret:
            print("Reached the end of the video or failed to read frame.")
            break


    # for i, frame in enumerate(frame_files):
    #     image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)



        # Check if the image was successfully loaded
        if image is None:
            print("Failed to load image")
        else:

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Convert grayscale image to BGR so we can plot colored stuff on it
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            h, w, _ = image.shape
            # Plot a keypoint
            kpts_x = int(kpts_yolo[i].bboxes[0].kpts['RWrist']['x']*w)
            kpts_y = int(kpts_yolo[i].bboxes[0].kpts['RWrist']['y']*h)
            # Plot a point (draw a small circle)
            cv2.circle(image, (kpts_x, kpts_y), radius=5, color=(0, 0, 255),
                       thickness=-1)  # Red point

            nose_x, nose_y = kpts_yolo[i].bboxes[0].kpts['Nose']['x']*w, kpts_yolo[i].bboxes[0].kpts['Nose']['y']*h
            draw_axis_3d(yaw=head_pose['yaw'].iloc[i], pitch=head_pose['pitch'].iloc[i], roll=head_pose['roll'].iloc[i], image=image, tdx=nose_x, tdy=nose_y)

            # 1300 for point,  2500 bottle
            image, projected_x,projected_y = estimate_visual_cone_on_table(size=1300, yaw=head_pose['yaw'].iloc[i], pitch=head_pose['pitch'].iloc[i], roll=head_pose['roll'].iloc[i], image=image, tdx=nose_x, tdy=nose_y)

            # Plot BBox
            # Draw a rectangle
            for bb in frames_bbox[i]:
                x= bb.bbox
            # x = frames_bbox[i].get_obj('person').bbox
                x = np.asarray([float(i) for i in x])
            # bbox[0].bboxes[0].bbox

                x1, y1, x2, y2= xywhn2xyxy(x,w,h)
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 100, 100), thickness=2)  # Blue rectangle
                cv2.putText(image, org=(x1,y2-10), text=bb.string_class, color=(255, 0, 128), fontFace=cv2.FONT_HERSHEY_TRIPLEX,thickness=2,fontScale=1)

            # Compute distances
            if target_is_point:
                target_detection = target_point
            else:
                target_detection = (frames_bbox[i].get_obj(target_obj_string))

            if intial_position is None:
                intial_position = float2pixel((target_detection.x1, target_detection.y1), w, h)


            if not target_detection.fake:
                target_temp = float2pixel((target_detection.x1, target_detection.y1), w, h)
                if follow or not detected:
                    target = target_temp
                    detected = True

                hand = kpts_yolo[i].bboxes[0].kpts['RWrist']['x']*w, kpts_yolo[i].bboxes[0].kpts['RWrist']['y']*h

                dist_head_target.append(dist_points(a=target, b=(projected_x, projected_y)))
                dist_hand_target.append(dist_points(a=target, b=hand))
            else:
                pass

            # plot target centroid
            cv2.circle(image, intial_position, radius=10, color=(0, 255, 255),
                       thickness=-1)  # YELLOW POINT
            cv2.circle(image, float2pixel((target_detection.x1, target_detection.y1), w, h), radius=5, color=(10, 10, 255),
                       thickness=-1) # RED POINT
            print(f'target: {(target_detection.x1, target_detection.y1)}')

            if len(dist_head_target)>0 and len(dist_hand_target)>0 and plot_on_screen:
                cv2.putText(image, org=(100, 100), text=f'head: {int(dist_head_target[-1])}, hand: {int(dist_hand_target[-1])}', color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            thickness=2, fontScale=1)


            # Display the image
            cv2.imshow(window_name, image)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if target_is_point:
                out_folder_video = Path(f'/home/federico/Data/Human_motion_videos/anticipation_out_videos/target_point/{action}/')
            else:
                out_folder_video = Path(f'/home/federico/Data/Human_motion_videos/anticipation_out_videos/target_reaching/{action}/')
            out_folder_video.mkdir(exist_ok=True, parents=True)
            file_name = f'{i}.jpeg'
            cv2.imwrite(str(out_folder_video / file_name), image)




            # Press 'q' to quit the video display

            # if cv2.waitKey(frame_delay) & 0xFF==ord('q'):
            #     break
            # Wait for a key press indefinitely or for a specified amount of time in milliseconds
            cv2.waitKey(0)
            i += 1

    # Close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    dist_head_target = [0 if x<20 else x for x in dist_head_target]
    data = {'x':list(range(len(dist_head_target))),'dist_head_target': dist_head_target, 'dist_hand_target': dist_hand_target}




    print(f'ciao')