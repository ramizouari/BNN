#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:11:41 2022

@author: ramizouari
"""

import tensorflow 
import larq

class BiRealBase(tensorflow.keras.layers.Layer):
    def __init__(self,estimator_type,*args,**kwargs):
        super(BiRealBase,self).__init__()
        self.batchNorm=tensorflow.keras.layers.BatchNormalization()
        self.estimator_type=estimator_type
        self.estimator_metadata={"args":args,"kwargs":kwargs}
    
    def build(self,input_shape):
        super(BiRealBase,self).build(input_shape)
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[1:],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        
        pass
    
    def build_bach_norm(self,*args,**kwargs):
        self.batchNorm=tensorflow.keras.layers.BatchNormalization(*args,**kwargs)
        return self
    
    def call(self,inputs,training=False):
        super(BiRealBase,self).call(inputs)
        return self.estimator(self.batchNorm(inputs))+inputs
        pass