import json
from pathlib import Path

import jsonpickle
import matplotlib
import pandas as pd
import torch

from kpts_data_base_load import extract_one_joint

if __name__ == '__main__':


    kpts_root_folder_data = Path('/home/federico/Data/Human_Motion')  # positions_3d_centered_shortened.csv

    my_dataset = pd.read_csv(kpts_root_folder_data / 'positions_3d_reformat.csv', header=[0])
    col = my_dataset.columns.values.tolist()
    my_data = extract_one_joint(my_dataset, 'rwrist')
    print(f'ciao')