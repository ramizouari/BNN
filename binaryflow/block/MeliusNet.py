#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:15:08 2022

@author: ramizouari
"""

import tensorflow
import larq


class Augmentation(tensorflow.keras.layers.Layer):
    def __init__(self,layer):
        self.layer=layer
        super(Augmentation,self).__init__()
        
    def call(self,inputs,training=False):
        return tensorflow.concat([inputs,self.layer.call(inputs,training)])

class AugmentationDense(Augmentation):
    def __init__(self,augment_by,dense_type=larq.layers.QuantDense,*args,**kwargs):
        layer=dense_type(*args,**kwargs)
        super(AugmentationDense,self).__init__(layer)
        
    def build(self,inputs_shape):
        pass
    
        
    
    
class Improvement(tensorflow.keras.layers.Layer):
    def __init__(self,layer):
        self.layer=layer
    def call(self,inputs,training=False):
        R=self.layer.call(inputs,training)
        A,B=tensorflow.split(inputs,[inputs.shape[-1]-R.shape[-1],R.shape[-1]])
        return tensorflow.concat([A,tensorflow.add(B,R)])



class Melius(tensorflow.keras.Sequential):
    def __init__(self,increase,improvement,*args,**kwargs):
        pass