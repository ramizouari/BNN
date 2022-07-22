# -*- coding: utf-8 -*-

import faulthandler; 
faulthandler.enable()
import os
import pickle
import numpy as np
import re

def load_batches(root:str,fileRegex:str,dtype=np.float):
    X=[]
    Y=[]
    for file in os.listdir(root):
        if re.search(fileRegex, file):
            U=pickle.load(open(f"{root}/{file}","rb"),encoding="bytes")
            X.append(U[b"data"])
            Y.append(U[b"labels"])
    X= np.array(X,dtype=dtype)
    Y=np.array(Y,dtype=dtype)
    return X,Y
    

def load_cifar(root:str,fileRegex:str,data_format="channels_last"):
    X,Y=load_batches(root,fileRegex)
    X=np.array(X).reshape((-1,3,32,32))
    Y=np.array(Y).reshape(-1,)
    if data_format == "channels_first":
        return X,Y
    elif data_format == "channels_last":
        return np.swapaxes(X,1,-1),Y
    else:
        raise RuntimeError(f"Format {data_format} is not recognized")