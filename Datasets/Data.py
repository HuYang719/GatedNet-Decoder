'''
Implementation of generate training and testing data
author: Lucyyang
'''
import pyldpc
import numpy as np

def encoding(k=8,N=16,H=None):
    if(k==8 and N == 16):
        H = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
    tG=pyldpc.CodingMatrix(H)
    return tG


# Create all possible information words
def dec2bin(num):
    l = np.zeros((8), dtype='int64')
    i = 7;
    while True:
        num, remainder = divmod(num, 2)
        l[i] = int(remainder)
        i = i - 1
        if num == 0:
            return l
# Generate the training data
def genData(k,N,num):
    tG = encoding(k, N, [])
    label = np.zeros((num, k), dtype='int64')
    for s in range(0, 256):
        label[s] = dec2bin(s)

    # Create sets of all possible codewords (codebook)
        data = np.zeros((num, N), dtype=int)
    for i in range(0, num):
        data[i] = (pyldpc.Coding(tG, label[i],0) + 1) / 2  # no Noise! HY：修改了pyLDPC源文件，pyLDPC生产码字不加噪声，统一用noise_layers层加噪
    data = data.reshape(-1, 16)
    return data,label

def genRanData(k,N,num,seedrand):
    np.random.seed(seedrand)
    tG = encoding(k, N, [])
    d_test = np.random.randint(0, 2, size=(num, k))
    x_test = np.zeros((num, N))
    for iii in range(0, num):
        x_test[iii] = (pyldpc.Coding(tG, d_test[iii], 2) + 1) / 2
    return x_test, d_test,


if __name__ == '__main__':
    k = 8
    N = 16
    num=256
    tG=encoding(8,16,[])
    data,label=genData(k,N,num)
    print('Data:',data.shape,data[0:5])
    print('Label:',label.shape,label[0:5])

