#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:30:23 2022

@author: ramizouari
"""
import tensorflow
from larq.layers import QuantDense,QuantConv1D,QuantConv2D,QuantConv3D
from binaryflow import deploy


class Dense(QuantDense):
    def __init__(self, units, **kwargs):
        super(Dense,self).__init__(units, **kwargs)
        self.binarized_kernel=None
        self.padding=None
        self.deployed=tensorflow.Variable(False,trainable=False)
        
        
    def deploy(self):
        binarized_kernel,self.padding=deploy.binarize_tensor(tensorflow.transpose(self.kernel),retPaddingCount=True)
        self.binarized_kernel.assign(binarized_kernel)
        self.deployed.assign(True) 
        return self
    
    def build(self,input):
        super(Dense,self).build(input)
        self.binarized_kernel=tensorflow.Variable(
            tensorflow.zeros(deploy.get_binarized_shape(shape=tensorflow.transpose(self.kernel).shape),dtype=tensorflow.uint64),
                                                  trainable=False)
        
    def call(self,input,is_training=False,**kwargs):
        return tensorflow.cond(self.deployed, 
                       lambda : tensorflow.cast
                       (
                           deploy.binarized_batch_mat_mul_tensor(self.binarized_kernel,input,transpose=False),
                           tensorflow.float32),
                       lambda :super(Dense,self).call(input))
            
            
            
    def call_deployed(self,input):
        return deploy.binarized_batch_mat_mul_tensor(self.binarized_kernel, input,transpose=False)
    
    
class DeployedDense(tensorflow.Module):
    def __init__(self,dense:Dense,name=None):
        super(DeployedDense,self).__init__(name)
        dense.binarized_kernel=tensorflow.transpose(deploy.binarize(dense.kernel))
    
    def __call__(self,input,is_training=False):
        return deploy.binarized_batch_mat_mul_tensor(self.binarized_kernel,input,transpose=False)