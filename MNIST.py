#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:50:55 2022

@author: ramizouari
"""

import tensorflow as tf
import binaryflow
from binaryflow.layers import ABCNet
from binaryflow.layers.normalization import ImageNormalizationLayer
from binaryflow import quantizers
import numpy as np


if __name__=="__main__":
    data_format="channels_last"
    
    (X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
    X_train=X_train.astype(dtype=np.float32)
    y_train=tf.one_hot(y_train,10)
    
    # All quantized layers except the first will use the same options
    
    kwargs = dict(
                  kernel_quantizers=quantizers.ShiftedSteSign,
                  input_quantizers=quantizers.ShiftedSteSign,
                  kernel_constraint="weight_clip",
                  kernel_params={"mu_initializer":tf.keras.initializers.RandomNormal(0,0.05)},
                  use_bias=False
                  )
    
    Dense=ABCNet.ABCDense
    Conv2D=ABCNet.ABCConv2D

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.GaussianNoise(stddev=4),
        ImageNormalizationLayer(),
        tf.keras.layers.BatchNormalization(momentum=0.999,scale=False),
        Dense(1024,activation="relu", **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999,scale=False),
        Dense(1024,activation="relu", **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999,scale=False),
        Dense(10, **kwargs),
        tf.keras.layers.Activation("softmax")
    ])
    
    
    model.compile(
        tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    
    trained_model = model.fit(
        X_train, 
        y_train,
        batch_size=128, 
        epochs=30,
        validation_data=(X_test, tf.one_hot(y_test, 10)),
        shuffle=True
    )