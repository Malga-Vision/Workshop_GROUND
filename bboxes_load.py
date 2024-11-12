'''Script defining detections and bounding box objects'''

import re
from pathlib import Path
from typing import Union

from coco_classes import COCO_CLASS, COCO_CLASS_str



def xywh_to_xyxy(x, y, w, h):
    """
    Convert bounding box from xywh format to xyxy format.

    Parameters:
    x (int): x coordinate of the top-left corner
    y (int): y coordinate of the top-left corner
    w (int): width of the bounding box
    h (int): height of the bounding box

    Returns:
    tuple: (x1, y1, w, h) coordinates of the bounding box
    """
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
def float2pixel(x, w,h):
    return int(w * x[0]), int(h * x[1])
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, w, h] where
            x1,y1 is the top-left corner, w,h is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y.int() if isinstance(y, torch.Tensor) else y.astype(int)

def load_bboxes(filename):
    with open(filename) as f:
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
            class_id, x1,y1,x2,y2, conf, id = bb.split(" ")
            bbox_list.append(Detections(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, track_id=id, coco_class=class_id))
        frame_list.append(Frame(frame_num=i, bbox=bbox_list))


    return frame_list

def load_frames_with_bboxes(person:int, action:str, trial:int):
    """

        :param person: in [0, 15]
        :param action: one out of ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle',
                   'touch_rubiks_cube',
                   'transport_bottle', 'transport_pen', 'transport_rubiks_cube']
        :param trial: 0 or 1
        """
    data_path= Path('/home/federico/Data/Human_motion_videos/Object_Detection')
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
            class_id, x1,y1,x2,y2, conf, id = bb.split(" ")
            bbox_list.append(Detections(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, track_id=id, coco_class=class_id))
        frame_list.append(Frame(frame_num=i, bbox=bbox_list))


    return frame_list


def split_by_numbered_lines(content):
    # Regex to match lines with only one number
    pattern = re.compile(r'^\d+$', re.MULTILINE)

    # Find all the positions of matches
    matches = list(pattern.finditer(content))

    # If no matches, return the whole content as one piece
    if not matches:
        return [content]

    # Split the content into pieces
    pieces = []
    start = 0
    for match in matches:
        end = match.start()
        piece = content[start:end].strip()
        if piece:
            pieces.append(piece)
        start = match.end()

    # Add the last piece of the content
    final_piece = content[start:].strip()
    if final_piece:
        pieces.append(final_piece)

    return pieces

class Detections:
    def __init__(self, x1, y1, x2, y2, coco_class, confidence=None, track_id=None):
        """Each detection is associated with a bounding box and a class.

        :param x1: bounding box 1st x coordinate
        :param y1: bounding box 1st y coordinate
        :param x2: bounding box 2nd x coordinate
        :param y2: bounding box 2nd y coordinate
        :param coco_class: class id according to COCO dataset
        :param confidence: classifier output
        :param track_id: id for the tracker
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.track_id = track_id
        self.bbox = (x1, y1, x2, y2)
        self.coco_class=coco_class
        self.coco_class_index=COCO_CLASS
    def __repr__(self):
        return f'Class:{self.coco_class}, id:{self.track_id}, bbox:{self.bbox}, conf:{self.confidence}'

    @property
    def centroid(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    @property
    def complete_bbox(self):
        return ((self.x1,self.y1),(self.x1,self.y2),(self.x2,self.y1),(self.x2,self.y2))

    @property
    def string_class(self):
        return COCO_CLASS[int(self.coco_class)]

class Frame:
    def __init__(self, frame_num, bbox:Union[Detections, list], *args, **kwargs):
        """Frame object, each one is characterised by some Detections object.

        :param frame_num: temporal id of the frmae
        :param bbox: list of bounding box coordinates or only one Detections objetc
        :param args: as many bboxes you want
        :param kwargs: unused
        """
        self.frame_num = frame_num
        if isinstance(bbox, Detections):
            self.bboxes = [bbox]
        elif isinstance(bbox, list):
            self.bboxes = bbox
        if len(args) > 0:
            self.bboxes.extend(args)
    def __repr__(self):
        return f"Frame with {self.detection_num} detections"

    @property
    def detection_num(self):
        return len(self.bboxes)

    @property
    def people_num(self):
        counter =0
        for bbox in self.bboxes:
            if bbox.string_class=='person':
                counter +=1
        return counter





if __name__ == '__main__':
    data_path= Path('./example_dir')

    actions = ['drinking', 'eat_crisp', 'open_close_bottle', 'rubiks_cube', 'sanitise', 'touch_bottle', 'touch_rubiks_cube',
     'transport_bottle', 'transport_pen', 'transport_rubiks_cube']
    # f'{person_folder.name}_{action_folder.name}_{trial_fodler.name}'

    # value in [0, 15]
    person=0
    # every value in actions
    action='Drinking'
    # 0 or 1
    trial=0
    if action.lower() not in actions:
        raise ValueError(f'Action {action.lower()} not recognized')
    # build the file name e.g. 00_transport_pen_0.txt
    file_name = str(person).zfill(2)+'_'+action.lower()+'_' + f'{trial}.txt'

    aa = load_bboxes(data_path/file_name)


    print(aa)