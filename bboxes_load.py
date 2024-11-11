import re
from pathlib import Path
from typing import Union

from coco_classes import COCO_CLASS, COCO_CLASS_str






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
        return ((x1,y1),(x1,y2),(x2,y1),(x2,y2))

    @property
    def string_class(self):
        return COCO_CLASS[int(self.coco_class)]

class Frame:
    def __init__(self, frame_num, bbox:Union[Detections, list], *args, **kwargs):
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