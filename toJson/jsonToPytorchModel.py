import sys
import os
import torch
import numpy as np
import json
sys.path.append('../src/')
from network import *

"""
load Json to pytorch model.

Json structure:

{
  "calc_freq": 1,
  "frame_length": 0.025,
  "shift_length": 0.01,
  "type":"dnn",
  "inputSize": [31, 26, 1],

  "layers":
    [
        {"layer_type": "fully_connect", "unit_num": 256, "weights": [[],[],...,[]], "bias": []},
        {"layer_type": "fully_connect", "unit_num": 256, "weights": [[],[],...,[]], "bias": []},
        {"outputSize": 2, "output_layer_weights": [[]], "output_layer_bias":[]}
    ]
}

__Author__ = Zhu.Bq
    
"""

jsonfile = 'vad128_128_SNR98_db0_25_RESCALE-45_-15_V1_20170825201833_epoch78_0.928171.json'
print('loading json from: %s' % jsonfile)
json_data = json.load(open(jsonfile, 'r'))

model = DNN_old()
# model = DNN()
print model

fc1_weights = torch.from_numpy(np.array(json_data['layers'][0]['weights'])).type(torch.FloatTensor)
fc1_bias = torch.from_numpy(np.array(json_data['layers'][0]['bias'])).type(torch.FloatTensor)
model.fc1.weight.data = fc1_weights
model.fc1.bias.data = fc1_bias

fc2_weights = torch.from_numpy(np.array(json_data['layers'][1]['weights'])).type(torch.FloatTensor)
fc2_bias = torch.from_numpy(np.array(json_data['layers'][1]['bias'])).type(torch.FloatTensor)
model.fc2.weight.data = fc2_weights
model.fc2.bias.data = fc2_bias


fc3_weights = torch.from_numpy(np.array(json_data['output_layer_weights'])).type(torch.FloatTensor)
fc3_bias = torch.from_numpy(np.array(json_data['output_layer_bias'])).type(torch.FloatTensor)
model.fc3.weight.data = fc3_weights
model.fc3.bias.data = fc3_bias

model.cuda()

modelname = '../model/model_old.pkl'
torch.save(model, modelname)
print('model has been saved as: %s' % modelname)

