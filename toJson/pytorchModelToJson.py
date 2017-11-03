
import sys
import torch
import numpy as np
import json
sys.path.append('../src/')

from network import *

"""
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

modelfile = '../model/dnn_2000h_v1_gpu06.pkl'
print('loading ' + modelfile + '...')

model = torch.load(modelfile)

weight1 = model.fc1.weight.data.cpu().numpy().tolist()
bias1 = model.fc1.bias.data.cpu().numpy().tolist()

weight2 = model.fc2.weight.data.cpu().numpy().tolist()
bias2 = model.fc2.bias.data.cpu().numpy().tolist()

weight3 = model.fc3.weight.data.cpu().numpy().tolist()
bias3 = model.fc3.bias.data.cpu().numpy().tolist()

layers = []

layer1 = {"layer_type": "fully_connect", "unit_num": 256}
layer1["bias"] = bias1
layer1["weight"] = weight1
layers.append(layer1)

layer2 = {"layer_type": "fully_connect", "unit_num": 256}
layer2["bias"] = bias2
layer2["weight"] = weight2
layers.append(layer2)

layer3 = {"outputSize": 2}
layer3["bias"] = bias3
layer3["weight"] = weight3
layers.append(layer3)

model_json = {"calc_freq": 1, "frame_length": 0.025, "shift_length": 0.01, "type":"dnn"}
model_json["inputSize"] = [31, 26, 1]
model_json['layers'] = layers

out = json.dumps(model_json, sort_keys=True)
out_file = open('./vad_dnn_2000h_epoch4_0.975.json', 'w')
out_file.write(out)
out_file.close()

print('Json file has been saved as vad_dnn_2000h_epoch4_0.975.json')
