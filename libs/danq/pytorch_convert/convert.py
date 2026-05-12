import h5py
from danq_pytorch import DanQ_
from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
from lstm import LSTM

def checkStateDict(model:nn.Module):
    print(model.state_dict())

def checkHDF5(wf_keras):
    with h5py.File(wf_keras, 'r') as f:
        for layer in f:
            print(layer)
            for group in f[layer]:
                weights = f[layer][group][:]
                print('    ' + group + ' ' + str(weights.shape))

def convertBias(weights:OrderedDict, f:h5py.File,
                  torch_layer:str,
                  keras_layer_id:int, kears_param_id:int=0):
    
    LID = keras_layer_id
    PID = kears_param_id

    weight = torch_layer + '.bias'
    print(weight)
    w_keras = f[f'layer_{LID}'][f'param_{PID}'][:]
    w_torch = weights[weight]
    w_torch[:] = torch.from_numpy(w_keras)
    print('keras', type(w_keras), w_keras.shape, w_keras.dtype)
    print('torch', type(w_torch), w_torch.shape, w_torch.dtype)

def convertLinear(weights:OrderedDict, f:h5py.File,
                  torch_layer:str,
                  keras_layer_id:int, kears_param_id:int=0):
    
    LID = keras_layer_id
    PID = kears_param_id

    weight = torch_layer + '.weight'
    print(weight)
    w_keras = f[f'layer_{LID}'][f'param_{PID+0}'][:]
    w_torch = weights[weight]
    w_keras = np.transpose(w_keras, (1, 0))
    w_torch[:] = torch.from_numpy(w_keras)
    print('keras', type(w_keras), w_keras.shape, w_keras.dtype)
    print('torch', type(w_torch), w_torch.shape, w_torch.dtype)

    convertBias(weights, f, torch_layer, LID, PID+1)

def convertConv1d(weights:OrderedDict, f:h5py.File,
                  torch_layer:str,
                  keras_layer_id:int, kears_param_id:int=0):
    
    LID = keras_layer_id
    PID = kears_param_id

    weight = torch_layer + '.weight'
    print(weight)
    w_keras = f[f'layer_{LID}'][f'param_{PID+0}'][:]
    w_torch = weights[weight]
    w_keras = w_keras[:,:,:,0]
    w_keras = np.flip(w_keras, axis=2).copy()
    w_torch[:] = torch.from_numpy(w_keras)
    print('keras', type(w_keras), w_keras.shape, w_keras.dtype)
    print('torch', type(w_torch), w_torch.shape, w_torch.dtype)

    convertBias(weights, f, torch_layer, LID, PID+1)

def convertLSTMGate(weights:OrderedDict, f:h5py.File,
                  torch_layer:str,
                  keras_layer_id:int, kears_param_id:int=0):
    
    LID = keras_layer_id
    PID = kears_param_id
    
    weight = torch_layer + '.weight'
    print(weight)
    w_keras = [
        f[f'layer_{LID}'][f'param_{PID+0}'][:].T,
        f[f'layer_{LID}'][f'param_{PID+1}'][:].T,
    ]
    w_torch = weights[weight]
    w_keras = np.hstack(w_keras)
    w_torch[:] = torch.from_numpy(w_keras)
    print('keras', type(w_keras), w_keras.shape, w_keras.dtype)
    print('torch', type(w_torch), w_torch.shape, w_torch.dtype)

    convertBias(weights, f, torch_layer, LID, PID+2)

def convertLSTM(weights:OrderedDict, f:h5py.File,
                  torch_layer:str,
                  keras_layer_id:int, kears_param_id:int=0):
    
    LID = keras_layer_id
    PID = kears_param_id

    convertLSTMGate(weights, f, torch_layer+'.W_i', LID, PID+0)
    convertLSTMGate(weights, f, torch_layer+'.W_c', LID, PID+3)
    convertLSTMGate(weights, f, torch_layer+'.W_f', LID, PID+6)
    convertLSTMGate(weights, f, torch_layer+'.W_o', LID, PID+9)

def convert(wf_keras:str, wf_torch:str,
            model_type:nn.Module, layers:list):

    model = model_type()
    weights = model.state_dict()

    with h5py.File(wf_keras) as f:
        for layer in layers:
            layer_type, torch_layer, keras_layer, keras_param = layer
            if layer_type == nn.Linear:
                convertLinear(weights, f, torch_layer, keras_layer, keras_param)
            elif layer_type == nn.Conv1d:
                convertConv1d(weights, f, torch_layer, keras_layer, keras_param)
            elif layer_type == LSTM:
                convertLSTM(weights, f, torch_layer, keras_layer, keras_param)
        
    torch.save(weights, wf_torch)
    return weights

def extract(wf_keras, wf_subkeras):
    # shutil.copy(wf_keras, wf_subkeras)
    layers = [
        'layer_0',
        'layer_1',
        'layer_2',
        'layer_3',
        'layer_4',
        # 'layer_5',
        # 'layer_6',
        # 'layer_7',
        # 'layer_8',
        # 'layer_9',
    ]
    with h5py.File(wf_keras, 'r') as dbi, h5py.File(wf_subkeras, 'r+') as dbo:
        for layer_i, layer in enumerate(layers):
            for ds in dbi[layer]:
                dbo[layer][ds][:] = dbi[layer][ds][:]
    
def loadData(path):
    with h5py.File(path, 'r') as db:
        data = db['testxdata']
        data = data[:]
    return data

def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return max(0, min(1, 0.2*x+0.5))

def tanh(x):
    return np.tanh(x)

if __name__ == '__main__':

    test_path = 'data/example.h5'

    convert(
        '/home/hyu/Digital_Platform/checks/DanQ/DanQ_bestmodel.hdf5',
        '/home/hyu/Digital_Platform/checks/DanQ/DanQ_bestmodel_pytorch.pth',
        DanQ_,
        [
            [nn.Conv1d, 'conv.0', 0, 0],
            [LSTM, 'biLSTM.forward_lstm.cell', 3, 0],
            [LSTM, 'biLSTM.backward_lstm.cell', 3, 12],
            [nn.Linear, 'fc1.0', 6, 0],
            [nn.Linear, 'fc2.0', 8, 0],
        ]
    )

