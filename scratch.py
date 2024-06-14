import json

import jsonpickle
import torch

# json.load('/home/federico/Videos/0000.json')

# with open('/home/federico/Videos/0000.json', 'r') as file:
#     data = json.load(file)
aaa = torch.load('/home/federico/Videos/0000.pt')

# recreated_obj = jsonpickle.decode(data)
aaa[0].show()