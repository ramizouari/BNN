#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:18:32 2022

@author: ramizouari
"""
import tensorflow as tf
import larq as lq
import numpy as np
import pandas as pd
import seaborn as sns
from binaryflow import quantizers
from binaryflow.layers import ABCNet,XnorNet,BinaryNet
from binaryflow.block import BiRealNet
from binaryflow.layers.normalization import *
from contextlib import redirect_stdout
import json
import matplotlib.pyplot as plt


data_format="channels_last"
    
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train=X_train.astype(dtype=np.float32)
y_train=tf.one_hot(y_train,10)
X_test=X_test.astype(dtype=np.float32)
X_train=X_train.reshape([*X_train.shape,1])
X_test=X_test.reshape([*X_test.shape,1])
# All quantized layers except the first will use the same options

abc_args = dict(
              kernel_quantizers=quantizers.ShiftedSteSign,
              input_quantizers=quantizers.ShiftedSteSign,
                kernel_estimators=5,
              kernel_constraint="weight_clip",
              kernel_params={"mu_initializer":tf.keras.initializers.RandomNormal(0,0.05)},
              use_bias=False
              )

bnn_args=dict(kernel_quantizer="ste_sign",
              input_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

Args=[bnn_args]*2+[abc_args]

Archs=[BinaryNet.QuantDense,XnorNet.ScaledQuantDense,ABCNet.ABCDense]
Names=["BinaryNet","XnorNet","ABCNet"]



Dense=BiRealNet.BiRealDense
Conv2D=XnorNet.ScaledQuantPlusConv2D



models=[]

x=tf.keras.layers.Input(shape=(28,28))
a=tf.keras.layers.GaussianNoise(stddev=4)(x)
a=ImageNormalizationLayer()(a)
#tf.keras.layers.BatchNormalization(momentum=0.999,scale=False),

#Conv2D(100,kernel=(3,3),mode=2,**bnn_args,activation="relu"),
a=tf.keras.layers.Flatten()(a)
a=tf.keras.layers.BatchNormalization(momentum=0.999,scale=False)(a)

a=BinaryNet.Dense(1024,activation="relu",**bnn_args)(a)
a=tf.keras.layers.BatchNormalization(momentum=0.999,scale=False)(a)
a=BinaryNet.Dense(1024,**bnn_args)(a)
a=tf.keras.layers.BatchNormalization(momentum=0.999,scale=False)(a)
a=tf.keras.layers.Dense(10)(a)
y=tf.keras.layers.Activation("softmax")(a)
model=tf.keras.Model(x,y)
models.append(model)
model.compile(
tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
loss="categorical_crossentropy",
metrics=["accuracy"],
)
model.run_eagerly=True

trained_model = model.fit(
    X_train, 
    y_train,
    batch_size=128, 
    epochs=1,
    validation_data=(X_test, tf.one_hot(y_test, 10)),
    shuffle=True
)


