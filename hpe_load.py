import json
from math import cos, sin
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

def draw_axis_3d(yaw, pitch, roll, image=None, tdx=None, tdy=None, size=50, yaw_uncertainty=-1, pitch_uncertainty=-1, roll_uncertainty=-1):
    """
    Draw yaw pitch and roll axis on the image if passed as input and returns the vector containing the projection of the vector on the image plane
    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z value that enlarge the "vector drawing"
            (default is 50)
    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    # print(yaw, pitch, roll)
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # PROJECT 3D TO 2D XY plane (Z = 0)
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdy
    if image is not None:
        cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
    return image

def estimate_visual_cone_on_table(yaw, pitch, roll, image=None, tdx=None, tdy=None, size=500, yaw_uncertainty=-1, pitch_uncertainty=-1, roll_uncertainty=-1, color=(255, 0, 0)):
    """
    Draw yaw pitch and roll axis on the image if passed as input and returns the vector containing the projection of the vector on the image plane
    Args:
        :yaw (float): value that represents the yaw rotation of the face
        :pitch (float): value that represents the pitch rotation of the face
        :roll (float): value that represents the roll rotation of the face
        :image (numpy.ndarray): The image where the three vector will be printed
            (default is None)
        :tdx (float64): x coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :tdy (float64): y coordinate from where the vector drawing start expressed in pixel coordinates
            (default is None)
        :size (int): value that will be multiplied to each x, y and z value that enlarge the "vector drawing"
            (default is 50)
    Returns:
        :list_projection_xy (list): list containing the unit vector [x, y, z]
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    # print(yaw, pitch, roll)
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # PROJECT 3D TO 2D XY plane (Z = 0)
    # # X-Axis pointing to right. drawn in red
    # x1 = size * (cos(yaw) * cos(roll)) + tdx
    # y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # # Y-Axis | drawn in green
    # x2 = size * (-cos(yaw) * sin(roll)) + tdx
    # y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    z3 = size * (cos(pitch) * cos(yaw)) + tdy
    if image is not None:
        # cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        # cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), color, 4)
    return image, x3,y3

def process_algo_2(data):
    temp_list = []
    for person in data['people']:
        temp_dict = {}
        for key, value in person.items():
            if key=='id_person':
                temp_dict[key] = value[0]
            elif key=='center_xy':
                temp_dict['tdx'] = value[0]
                temp_dict['tdy'] = value[1]
            else:
                temp_dict[key] = value[0]
        temp_list.append(temp_dict)
    return temp_list

def load_hpe(person:int, action:str, trial:int):
    """

    :param person: in [0, 15]
    :param action: one out of ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
               'touch_rubiks_cube',
               'transport_bottle', 'transport_pen', 'transport_rubiks_cube']
    :param trial: 0 or 1
    """
    root_folder = Path('/home/federico/Data/Human_motion_videos/head_pose_estimation_centernet/')
    total_dataframe = pd.DataFrame(
            columns=['id_person', 'yaw', 'pitch', 'roll', 'yaw_u', 'pitch_u', 'roll_u', 'tdx', 'tdy', 'frame'])

    person = person
    if person <0 or person > 15:
        raise ValueError('person should be in [0, 15]')
    action = action
    actions = ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
               'touch_rubiks_cube',
               'transport_bottle', 'transport_pen', 'transport_rubiks_cube']

    if action.lower() not in actions:
        raise ValueError(f'Action {action.lower()} not recognized')
    trial = trial
    if trial < 0 or trial > 1:
        raise ValueError('trial should be or 0 or 1')

    target_directory = root_folder / str(person).zfill(2) / action.lower() / f'{trial}'

    video_dataframe = pd.DataFrame(
            columns=['id_person', 'yaw', 'pitch', 'roll', 'yaw_u', 'pitch_u', 'roll_u', 'tdx', 'tdy',
                     'frame'])
    list_of_frames = []
    for ypr_file in target_directory.iterdir():
        with open(ypr_file, 'r') as f:
            data = json.load(f)
            frame_list = process_algo_2(data)

            frame_dataframe = pd.DataFrame.from_dict(frame_list)
            frame_dataframe['frame'] = ypr_file.stem
            list_of_frames.append(frame_dataframe)
    # video_dataframe = video_dataframe.append(frame_dataframe, ignore_index=False)
    video_dataframe = pd.concat(list_of_frames)
    video_dataframe.sort_values(by=['frame', 'id_person'], inplace=True)
    # video_dataframe.to_csv(f'{csv_folder/video.stem}.csv')
    return video_dataframe



if __name__ == '__main__':

    out = load_hpe(person=0, action='drinking', trial=0)



    print(f'Ciao')