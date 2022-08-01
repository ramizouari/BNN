#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:15:20 2022

@author: ramizouari
"""

import BiRealBase
import larq

class BiRealDenseBlock(BiRealBase):
    def __init__(self,estimator_type=larq.layers.QuantDense,*args,**kwargs):
        self.estimator_type=estimator_type
        super(BiRealDenseBlock, self).__init__(estimator_type,*args,**kwargs)
    
    def build(self,input_shape):
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[-1],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        pass
    