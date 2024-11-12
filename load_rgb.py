from pathlib import Path

def load_rgb_video_path(person:int, action:str, trial:int):
    data_path = Path('/home/federico/Data/Human_motion_videos/output_videos')
    # f'{person_folder.name}_{action_folder.name}_{trial_fodler.name}'
    person = person
    if person < 0 or person > 15:
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
    target_directory = data_path / str(person).zfill(2) / action.lower() / f'{trial}'
    return target_directory/'video.mp4'
def load_rgb_name(person:int, action:str, trial:int):
    #/home/federico/Data/Human_motion_videos/output_videos/00/Drinking/0/frames/
    data_path= Path('/home/federico/Data/Human_motion_videos/output_videos')
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
    target_directory = data_path / str(person).zfill(2) / action.lower() / f'{trial}'/'frames'
    frame_list=[]
    for frame in target_directory.iterdir():
        frame_list.append(frame)

    frame_list.sort()
    return frame_list

if __name__ == '__main__':
    aa = load_rgb_name(person=1, action='drinking', trial=1)
    print('ciao')