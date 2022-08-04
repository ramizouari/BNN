#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:12:49 2022

@author: ramizouari
"""

import tensorflow 
import larq

class BiRealBase(tensorflow.keras.Sequential):
    def __init__(self,estimator_type,layers=None,name=None,*args,**kwargs):
        super(BiRealBase,self).__init__(layers,name)
        self.estimator_type=estimator_type
        self.estimator_metadata={"args":args,"kwargs":kwargs}
    
    def build(self,input_shape):
        self.add(self.estimator_type(input_shape[1:],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"]))
        super(BiRealBase,self).build(input_shape)
        pass
    
    def add(self,*args,**kwargs):
        super(BiRealBase,self).add(*args,**kwargs)
        return self
    
    def call(self,inputs,training=False):
        return tensorflow.add(super(BiRealBase,self).call(inputs),inputs)
        pass
    

class BiRealDense(BiRealBase):
    def __init__(self,estimator_type=larq.layers.QuantDense,*args,**kwargs):
        super(BiRealDense, self).__init__(estimator_type,*args,**kwargs)
        
    @staticmethod
    def WithBatchNorm(estimator_type=larq.layers.QuantDense,*args,**kwargs):
        return BiRealDense(estimator_type,layers=[
            tensorflow.kerasl.layers.BatchNormalization()],*args,**kwargs)
    
    def build(self,input_shape):
        self.add(self.estimator_type(input_shape[-1],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"]))
        pass
    


class BiRealConv2D(BiRealBase):
    def __init__(self,kernel,estimator_type=larq.layers.QuantConv2D,*args,**kwargs):
        super(BiRealConv2D, self).__init__(estimator_type,*args,**kwargs)
        self.kernel_shape=kernel
    
    def build(self,input_shape):
        self.add(self.estimator_type(input_shape[-1],self.kernel_shape,*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"]))
        pass
     