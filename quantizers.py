#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:11:33 2022

@author: ramizouari
"""

import larq
import tensorflow

class ShiftedQuantizer(larq.quantizers.Quantizer):
    def __init__(self,quantizer:larq.quantizers.Quantizer,mu=None,mu_initializer=None,*args,**kwargs):
        super(ShiftedQuantizer,self).__init__(*args,**kwargs)
        if isinstance(quantizer,type):
            self.quantizer=quantizer()
        else:
            self.quantizer=quantizer
        if mu is None:
            initializer=tensorflow.keras.initializers.RandomUniform() if mu_initializer is None else mu_initializer
            self.mu=tensorflow.Variable(initializer(shape=()),trainable=self.trainable)
        else:
            self.mu=tensorflow.Variable(float(mu))
            
    def build(self,inputs_shape):
        super(ShiftedQuantizer,self).build(inputs_shape)
    
    def call(self,inputs):
        return super(ShiftedQuantizer,self).call(tensorflow.add(inputs,self.mu))
    
    
class ShiftedSteSign(ShiftedQuantizer):
    precision=1
    def __init__(self,mu=None,mu_initializer=None,*args,**kwargs):
        super(ShiftedSteSign,self).__init__(larq.quantizers.SteSign,mu=mu,mu_initializer=mu_initializer,*args,**kwargs)
        
