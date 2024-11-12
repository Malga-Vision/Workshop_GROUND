import re
from pathlib import Path

import numpy as np
import torch

from bboxes_load import split_by_numbered_lines, Detections, Frame

coco_kpts = {
    0: "Nose",
    1: "LEye",
    2: "REye",
    3: "LEar",
    4: "REar",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle"
}
coco_kpts_strings={value: int(key) for key, value in coco_kpts.items()}

class Keypoints:
    def __init__(self, keypoints):
        self.xy = keypoints
        self.num_keypoints = len(self.keypoints)
        self.x = keypoints[:, 0]
        self.y = keypoints[:, 1]

    def xyn2xy(self, w, h):
        """Returns x, y coordinates from normalized xyn coordinates."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] *= w
        xy[..., 1] *= h
        return xy.int() if isinstance(xy, torch.Tensor) else xy.astype(int)

def load_frames_with_bboxes_kpts(person:int, action:str, trial:int):
    """

        :param person: in [0, 15]
        :param action: one out of ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
                   'touch_rubiks_cube',
                   'transport_bottle', 'transport_pen', 'transport_rubiks_cube']
        :param trial: 0 or 1
        """
    data_path= Path('/home/federico/Data/Human_motion_videos/kpts_yolo')
    # f'{person_folder.name}_{action_folder.name}_{trial_fodler.name}'
    person=person
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
    filename = str(person).zfill(2)+'_'+action.lower()+'_' + f'{trial}.txt'

    with open(data_path/filename) as f:
        data = f.read()

    # Compile the regex pattern
    pattern = re.compile(r'^\d+$', re.MULTILINE)

    # Find all matches using finditer
    matches = pattern.finditer(data)

    # Print the matches
    for match in matches:
        print(f'Found number: {match.group()} at position {match.start()} to {match.end()}')


    # Split the content into pieces
    pieces = split_by_numbered_lines(data)

    frame_list = []

    for i, p in enumerate(pieces):
        bbox_list = []
        for bb in p.split('\n'):
            temp = np.asarray(bb.split(" ")).astype(float)
            class_id, x1, y1, w, h, keypoints = int(temp[0]), temp[1], temp[2], temp[3], temp[4], temp[5:-2]
            triplets = keypoints.reshape(-1,3)
            # coco_kpts_list_labels = coco_kpts_strings.
            sorted_kpts = [coco_kpts[key] for key in sorted(coco_kpts.keys())]
            nested_dict = {label: {'x': triplet[0], 'y': triplet[1], 'z': triplet[2]} for label, triplet in zip(sorted_kpts, triplets)}
            bbox_list.append(Detections(x1=x1, y1=y1, x2=w, y2=h, track_id=id, coco_class=class_id, keypoints=nested_dict))
        frame_list.append(Frame(frame_num=i, bbox=bbox_list))


    return frame_list

if __name__ == '__main__':
    aa = load_frames_with_bboxes_kpts(person=0, action='drinking', trial=0)

    print(aa)