#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 2022

@author: ramizouari
"""

import larq
import tensorflow
import tensorflow_probability



class ShiftedQuantizer(larq.quantizers.Quantizer):
    """
        ShiftedQuantizer is the base class of "shifted quantizers" denoted SQ
        It takes a quantizer Q, and a shifting parameter µ to create a shifted version of Q by µ
        
        <ul>
        <li> <strong>input</strong>: x, any tensor </li>
        <li> <strong>output</strong>: SQ(x)=Q(x+µ) </li>
        <li> <strong>parameters</strong>: µ => learnable by default
        </ul>
    """
    def __init__(self,quantizer,mu=None,mu_initializer=None,trainable=True,*args,**kwargs):
        super(ShiftedQuantizer,self).__init__(trainable=trainable,*args,**kwargs)
        if isinstance(quantizer,type):
            self.quantizer=quantizer()
        else:
            self.quantizer=quantizer
        if mu is None:
            initializer=tensorflow.keras.initializers.RandomUniform() if mu_initializer is None else mu_initializer
            self.mu=tensorflow.Variable(initializer(shape=()),trainable=trainable)
        else:
            self.mu=tensorflow.Variable(float(mu),trainable=trainable)
            
    def build(self,inputs_shape):
        super(ShiftedQuantizer,self).build(inputs_shape)
        self.quantizer.build(inputs_shape)
    
    def call(self,inputs):
        super(ShiftedQuantizer,self).call(inputs)
        return self.quantizer.call(tensorflow.add(inputs,self.mu))
    
    

class ShiftedSteSign(ShiftedQuantizer):
    """
        ShiftedSteSign: this is an implementation of the shifted sign function
        <ul>
        <li> <strong>input</strong>: x, any tensor </li>
        <li> <strong>output</strong>: SSS(x)=sign(x+µ) </li>
        <li> <strong>parameters</strong>: µ => learnable by default
        </ul>
    """
    precision=1
    def __init__(self,mu=None,mu_initializer=None,trainable=True,*args,**kwargs):
        super(ShiftedSteSign,self).__init__(larq.quantizers.SteSign,mu=mu,mu_initializer=mu_initializer,
                                            trainable=trainable,*args,**kwargs)



class StochasticSteSign(larq.quantizers.SteSign):
    def __init__(self,
                distribution:tensorflow_probability.distributions.Distribution=tensorflow_probability.distributions.Uniform(-1,1),
                *args,**kwargs):
        self.distribution=distribution
        super(StochasticSteSign,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        return super(StochasticSteSign,self).call(tensorflow.subtract(inputs,self.distribution.sample([1]+inputs.shape[1:])))

class NormalStochasticSteSign(StochasticSteSign):
    def __init__(self,stddev):
        super(NormallyStochasticSteSign,self).__init__(distribution=tensorflow_probability.distributions.Normal(0,stddev))

class UniformStochasticSteSign(StochasticSteSign):
    def __init__(self,a=-1.,b=1.):
        super(UniformStochasticSteSign,self).__init__(distribution=tensorflow_probability.distributions.Uniform(a,b))
        
class LaplaceStochasticSteSign(StochasticSteSign):
    def __init__(self,scale=0.5):
        super(LaplaceStochasticSteSign,self).__init__(distribution=tensorflow_probability.distributions.Laplace(0,scale))
#%%
