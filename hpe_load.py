import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def process_algo_2(data):
    """Function only to change data format.

    :param data: data loaded from the head pose estimator output.
    :return: the new data format, a list of people, each with its dictionary containign the data.
    """
    temp_list = []
    for person in data['people']:
        temp_dict = {}
        for key, value in person.items():
            if key=='id_person':
                temp_dict[key] = value[0][0]
            elif key=='center_xy':
                temp_dict['tdx'] = value[0]
                temp_dict['tdy'] = value[1]
            else:
                temp_dict[key] = value[0]
        temp_list.append(temp_dict)
    return temp_list


if __name__ == '__main__':
    angle = 'pitch'
    root_folder = Path('/home/federico/Data/Human_motion_videos/head_pose_estimation/')
    # main_folder = root_folder / 'Data'
    # main_folder = root_folder / 'Training_data_hpe/'


    # csv_folder = root_folder / 'Data_csv'
    # images_folder = root_folder/'Data_images_pitch'


    total_dataframe = pd.DataFrame(
        columns=['id_person', 'yaw', 'pitch', 'roll', 'yaw_u', 'pitch_u', 'roll_u', 'tdx', 'tdy', 'frame'])

    person = 0
    action = 'Drinking'
    trial = 0


    # for person in root_folder.iterdir():
    #     for action in person.iterdir():
    #         for trial in action.iterdir():
    actions = ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
               'touch_rubiks_cube',
               'transport_bottle', 'transport_pen', 'transport_rubiks_cube']

    if action.lower() not in actions:
        raise ValueError(f'Action {action.lower()} not recognized')

    target_directory = root_folder / str(person).zfill(2) /action.lower()/ f'{trial}'

    video_dataframe = pd.DataFrame(
        columns=['id_person', 'yaw', 'pitch', 'roll', 'yaw_u', 'pitch_u', 'roll_u', 'tdx', 'tdy',
                 'frame'])
    list_of_frames = []
    # List over Head Pose Estimation files (Head Pose Estimation done in advance)
    for ypr_file in target_directory.iterdir():
        # Open the json file containing a dictionary with a list of 'people' and a dictionary with the data, e.g.
        # {'center_xy': [pixel, pixel], 'id_person': [[-1]], 'pitch': [float_degree], 'pitch_u': [float], 'roll': [float_degree], 'roll_u': [float], 'yaw': [float_degree], 'yaw_u': [float]}
        with open(ypr_file, 'r') as f:
            data = json.load(f)
            frame_list = process_algo_2(data)


            frame_dataframe = pd.DataFrame.from_dict(frame_list)
            # add frame number
            frame_dataframe['frame'] = ypr_file.stem
            list_of_frames.append(frame_dataframe)
# video_dataframe = video_dataframe.append(frame_dataframe, ignore_index=False)
    video_dataframe = pd.concat(list_of_frames)
    # it is a vide, let it order by frame number and inside each of them by people in it.
    video_dataframe.sort_values(by=['frame', 'id_person'], inplace=True)
    # save if needed
    # video_dataframe.to_csv(f'{csv_folder/video.stem}.csv')

    print(f'Ciao')