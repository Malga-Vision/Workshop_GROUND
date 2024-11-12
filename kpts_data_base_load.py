"Primitive function for current_config loading and extraction and current_config creation"
from pathlib import Path

import pandas as pd
from ultralytics.utils import instance


def extract_one_joint(data: pd.DataFrame, joint: str):
    joint_x = joint + ' x'
    joint_y = joint + ' y'
    joint_z = joint + ' z'
    ret = data[["person", "action", "sample", "frame", joint_x, joint_y, joint_z]]
    return ret


def extract_one_action(data: pd.DataFrame, action: str):
    # check if action is present in dataset
    ret = data[data["action"] == action]
    return ret


def extract_one_person(data: pd.DataFrame, instance: int):
    ret = data[data["person"] == instance]
    return ret


def extract_one_video(data: pd.DataFrame, instance: int):
    if instance == 0 or instance == 1:
        ret = data[data["sample"] == instance]
        return ret
    else:
        return IndexError

def extract_one_action_sample(person:int, action: str, trial:int,joint:str='rwrist'):
    """

        :param person: in [0, 15]
        :param action: one out of ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
                   'touch_rubiks_cube',
                   'transport_bottle', 'transport_pen', 'transport_rubiks_cube']
        :param trial: 0 or 1
        """
    kpts_root_folder_data = Path('/home/federico/Data/Human_Motion')  # positions_3d_centered_shortened.csv
    my_dataset = pd.read_csv(kpts_root_folder_data / 'positions_3d_reformat.csv', header=[0])
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
    temp_data = extract_one_joint(data=my_dataset, joint=joint)
    temp_data= extract_one_person(data=temp_data, instance=person)
    temp_data = extract_one_action(data=temp_data, action=action)
    temp_data = extract_one_video(data=temp_data, instance=trial)
    return return_only_data(temp_data)


def return_only_data(data: pd.DataFrame):
    col = data.columns
    intersection = set(col) & {'person', 'action', 'sample', 'frame'}

    try:
        ret = data.drop(columns=list(intersection))
    except KeyError:
        pass
    return ret


if __name__ == '__main__':
    root_folder_data = Path('/home/federico/Data/Human_Motion') # positions_3d_centered_shortened.csv
    # input_path = root_folder_data / 'human_motion_dataset.parquet.gz'
    # Index(['individual', 'action', 'sample', 'current_config type', 'model', 'detection', 'joint', 'frame', 'variable', 'value', 'normalised_frame'],

    # my_dataset = pd.read_parquet(input_path)
    # my_dataset = pd.read_csv(root_folder_data / 'positions_3d_reformatted.csv', header=[0, 1, 2])
    my_dataset = pd.read_csv(root_folder_data / 'positions_3d_reformat.csv', header=[0])
    # a = my_dataset.loc["000000", :, :, ["head_pose", "key_points_2d"], :].unstack(level=[DATA, VARIABLE, JOINT])
    col = list(my_dataset.columns.values)
    my_data = extract_one_joint(my_dataset, 'rwrist')
    # my_dataset[my_dataset["Age"] > 35]
    print('e ora?')