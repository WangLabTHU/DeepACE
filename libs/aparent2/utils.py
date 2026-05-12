import numpy as np


def seq_2_onehot_aparent2(seqs):
    result = []
    for seq in seqs:
        bases = 'ACGT'
        onehot = np.zeros((len(seq), 4), dtype=int)
        for i, base in enumerate(seq):
            if base in bases:
                onehot[i, bases.index(base)] = 1
            elif base == 'N':
                pass
        onehot = onehot[None, :, :]
        result.append(onehot)    
    return np.array(result)


def onehot_2_seq_aparent2(onehots):
    bases = 'ACGT'
    seqs = []
    for onehot in onehots:
        tmp = []
        for i in range(onehot.shape[0]):
            if np.any(onehot[i]):
                j = np.argmax(onehot[i])
                tmp.append(bases[j])
            else:
                tmp.append("N")
        seq = ''.join(tmp)
        seqs.append(seq)
    return seqs

def padding_and_cropping_aparent2(seqs_encode, seq_len, input_len, model_name):
    
    shape = seqs_encode.shape
    axis_to_pad_crop = 2 # (2, 1, 205, 4)
    
    if seq_len < input_len:
        print("Start padding sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        padding_length = input_len - seq_len
        pad_left = padding_length // 2
        pad_right = padding_length - pad_left
        padding_dims = [(0, 0) if i != axis_to_pad_crop else (pad_left, pad_right) for i in range(len(shape))]
        seqs_encode = np.pad(seqs_encode, padding_dims, 'constant', constant_values=0)

    elif seq_len > input_len:
        print("Start cropping sequences to model settings of {}: from {} to {}".format(model_name, seq_len, input_len))
        cropping_length = seq_len - input_len
        crop_left = cropping_length // 2
        crop_right = cropping_length - crop_left
        
        slices = [slice(None)] * len(shape)
        slices[axis_to_pad_crop] = slice(crop_left, -crop_right if crop_right else None)
        seqs_encode = seqs_encode[tuple(slices)]
    return seqs_encode


def open_fa(file):
    record = []
    f = open(file,'r')
    for item in f:
        if '>' not in item:
            record.append(item[0:-1])
    f.close()
    return record

def write_txt(file, data):
    f = open(file,'w')
    i = 0
    while i < len(data):
        f.write(data[i] + '\n')
        i = i + 1
    f.close()