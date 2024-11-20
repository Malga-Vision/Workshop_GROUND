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
from bboxes_load import load_frames_with_bboxes, xywh_to_xyxy, xywhn2xyxy, float2pixel
from hpe_load import load_hpe, draw_axis_3d, estimate_visual_cone_on_table
from kpts_data_base_load import extract_one_joint, extract_one_action_sample
from load_kpts_yolo import load_frames_with_bboxes_kpts


def dist_points(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


if __name__ == '__main__':
    # actions = ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
    #            'touch_rubiks_cube',
    #            'transport_bottle', 'transport_pen', 'transport_rubiks_cube']

    for person in range(0,16):
        # person = 2
        print('\n')
        print(f'Person {person}')
        trial=1
        action = 'transport_bottle'
        target_obj_string='bottle'
        target_obj_string_to_write = 'bottle'
        follow=False
        plot_on_screen = True
        target_is_point = True
        Point = namedtuple('Point', ['x1', 'y1', 'fake'])
        target_point = Point(0.907543, 0.749613, False)


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
        print(f'i_max: {i_max}')
        # Open the video file
        cap = cv2.VideoCapture(video_file)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        window_name= ' Federico'
        # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

        # SET VARIABLE FOR SAVING
        dist_hand_target =[]
        dist_head_target =[]
        detected =False
        target =0,0


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
                h, w, _ = image.shape
                # Plot a keypoint
                kpts_x = int(kpts_yolo[i].bboxes[0].kpts['RWrist']['x']*w)
                kpts_y = int(kpts_yolo[i].bboxes[0].kpts['RWrist']['y']*h)
                # Plot a point (draw a small circle)
                cv2.circle(image, (kpts_x, kpts_y), radius=5, color=(0, 0, 255),
                           thickness=-1)  # Red point

                nose_x, nose_y = kpts_yolo[i].bboxes[0].kpts['Nose']['x']*w, kpts_yolo[i].bboxes[0].kpts['Nose']['y']*h
                # draw_axis_3d(yaw=head_pose['yaw'].iloc[i], pitch=head_pose['pitch'].iloc[i], roll=head_pose['roll'].iloc[i], image=image, tdx=nose_x, tdy=nose_y)

                image, projected_x,projected_y = estimate_visual_cone_on_table(size=1300, yaw=head_pose['yaw'].iloc[i], pitch=head_pose['pitch'].iloc[i], roll=head_pose['roll'].iloc[i], image=image, tdx=nose_x, tdy=nose_y)

                # Plot BBox
                # Draw a rectangle
                for bb in frames_bbox[i]:
                    x= bb.bbox
                # x = frames_bbox[i].get_obj('person').bbox
                    x = np.asarray([float(i) for i in x])
                # bbox[0].bboxes[0].bbox

                    x1, y1, x2, y2= xywhn2xyxy(x,w,h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # Blue rectangle
                    cv2.putText(image, org=(x1,y2), text=bb.string_class, color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=2,fontScale=1)
                # Display the image
                # cv2.imshow('Federico', image)
                # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


                # Compute distances
                if target_is_point:
                    target_detection = target_point
                else:
                    target_detection = (frames_bbox[i].get_obj(target_obj_string))
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

                # Press 'q' to quit the video display

                # if cv2.waitKey(frame_delay) & 0xFF==ord('q'):
                #     break
                # Wait for a key press indefinitely or for a specified amount of time in milliseconds
                # cv2.waitKey(0)
                i += 1

        # Close all OpenCV windows
        cap.release()
        # cv2.destroyAllWindows()

        dist_head_target = [0 if x<20 else x for x in dist_head_target]
        data = {'x':list(range(len(dist_head_target))),'dist_head_target': dist_head_target, 'dist_hand_target': dist_hand_target}


        from scipy import signal
        # take the beginning to avoid the second part, non vale se segui oggetto
        if len(data['dist_head_target'])>25:
            dist_head_smooth = signal.savgol_filter(data['dist_head_target'],
                                 25,  # window size used for filtering
                                 3)

            contact_frame_hand=np.argmin(np.asarray(dist_hand_target)[:100])
            contact_frame_head=np.argmin(np.asarray(dist_head_smooth)[:100])
            difference_contact= contact_frame_head - contact_frame_hand
            print(f'Difference in frames: {difference_contact}')
            if target_is_point:
                file_name=f'/home/federico/Data/Human_motion_videos/workshop_images/target_point/results_{action}_trial_{trial}.txt'
            else:
                file_name = f'/home/federico/Data/Human_motion_videos/workshop_images/target_reaching/results_{action}_trial_{trial}.txt'
            with open(file_name, 'a') as f:
                f.write(f'Person {person}, difference:  ')
                f.write(f"{difference_contact}\n")

            import plotly.express as px
            import plotly.graph_objects as go



            # Create traces
            fig = go.Figure()
            # fig.add_trace(go.Scatter(x=data['x'], y=data['dist_head_target'],
            #                          mode='lines',
            #                          name='head'))
            fig.add_trace(go.Scatter(x=data['x'], y=data['dist_hand_target'],
                                     mode='lines',
                                     name='Hand',
                                     line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=data['x'], y = dist_head_smooth,  # order of fitted polynomial,
                                     mode='lines',
                                     name='Head direction',
                                     line=dict(color='#EF553B')))
            fig.add_vline(x=contact_frame_hand, line_width=3, line_dash="dash", line_color='#636EFA') # , line_color="green"
            fig.add_vline(x=contact_frame_head, line_width=3, line_dash="dash", line_color='#EF553B')  # , line_color="green"

            # fig.update_layout(
            #         font_family="Courier New",
            #         font_color="blue",
            #         title_font_family="Times New Roman",
            #         title_font_color="red",
            #         legend_title_font_color="green"
            #         )
            # fig.update_xaxes(title_font_family="Arial")
            fig.update_layout(
                    title=f"Distance vs Frames <br><sup>Target is {target_obj_string_to_write}</sup>",
                    xaxis_title="Frames (time)",
                    yaxis_title="Distance in pixels",
                    legend_title="Legend",
                    font=dict(
                            family="Courier New, monospace",
                            size=18,
                            color="RebeccaPurple"
                            )
                    )



            # fig.show()
            if target_is_point:
                output_path = Path(f'/home/federico/Data/Human_motion_videos/workshop_images/target_point/{action}')
            else:
                output_path = Path(f'/home/federico/Data/Human_motion_videos/workshop_images/target_reaching/{action}')
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_image(output_path /f"{person}_{trial}.pdf")
        elif len(data['dist_hand_target'])<5:
            pass