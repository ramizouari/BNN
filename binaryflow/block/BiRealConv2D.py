#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:17:36 2022

@author: ramizouari
"""

import BiRealBase

class BiRealConv2D(BiRealBase):
    def __init__(self,kernel,estimator_type=larq.layers.QuantConv2D,*args,**kwargs):
        self.estimator_type=estimator_type
        super(BiRealConv2DBlock, self).__init__(estimator_type,*args,**kwargs)
        self.kernel_shape=kernel
    
    def build(self,input_shape):
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[-1],self.kernel_shape,*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        pass
     