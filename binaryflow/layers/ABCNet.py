#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:31:57 2022

@author: ramizouari
"""


import tensorflow
from .. import quantizers
from .XnorNet import ScaledQuantDense,ScaledQuantConv1D,ScaledQuantConv2D,ScaledQuantConv3D
from .. import quantizers



class ABCBase(tensorflow.keras.Model):
    """
    This is the base class of ABCNetworks
    It contains the functions shared by any ABCNet
    <ul>
        <li> <strong> hyperparameters </strong>:
            <ul>
                <li> kernel_estimators:  number of kernel quantizers</li>
                <li> input_estimators: number of input quantizers</li>
                <li> kernel_quantizers: the quantizers used for the kernel. defaults to ShiftedSteSign </li>
                <li> input_quantizers: the quantizers used for the input. defaults to ShiftedSteSign </li>
                <li> estimators: the estimators used to estimate the output of the non-binarised version of the layer. The content of estimators should be of the same class, and should contain the parameters of the underlying models </li>
            </ul>
        <li> <strong>input</strong>: x a tensor, with adequate shape (see subclasses) </li>
        <li> <strong>output</strong>: ABC(x)
    </ul>
    """
    def __init__(self,estimators,*args,**kwargs):
        super(ABCBase,self).__init__()
        self.kernel_estimators=len(estimators)
        self.estimators=estimators
    pass
    
    def build(self,input_shape):
        #self.kernels=tensorflow.Variable()
        for estimator in self.estimators:
            estimator.build(input_shape)
        #self.inputs=tensorflow.keras.layers.Input(shape=(input_shape[1:]))
        #self.outputs=self.call(self.inputs)
        
        
    def call(self,inputs,training=False):
        return tensorflow.add_n([estimator(inputs) for estimator in self.estimators])
    
    
    @staticmethod
    def get_quantizers(kernel_estimators,kernel_quantizers,input_quantizers,kernel_params,input_params):
        """
        get_quantizers is helper method that converts the kernel and input quantizers so that
        they can be easily constructed in any inherited class
        """
        if kernel_params is None:
            kernel_params={}
        if input_params is None:
            input_params={}
        if not isinstance(kernel_quantizers,(list,tuple)):
            kernel_quantizers=[kernel_quantizers]*kernel_estimators
        else:
            kernel_estimators=len(kernel_quantizers)
        
        if not isinstance(kernel_params,(list,tuple)):
            kernel_params=[kernel_params]*kernel_estimators
            
        if not isinstance(input_quantizers,(list,tuple)):
            input_quantizers=[input_quantizers]*kernel_estimators
        
        if not isinstance(input_params,(list,tuple)):
            input_params=[input_params]*kernel_estimators
        ker_quantizers=[]
        for kernel_quantizer,kernel_params in zip(kernel_quantizers,kernel_params):
            ker_quantizers.append(kernel_quantizer(**kernel_params) if isinstance(kernel_quantizer,type) else kernel_quantizer)
            
        in_quantizers=[]
        for input_quantizer,input_params in zip(input_quantizers,input_params):
            in_quantizers.append(input_quantizer(**input_params) if isinstance(input_quantizer,type) else input_quantizer)
            
        return ker_quantizers,in_quantizers
    
    
class ABCConvND(ABCBase):
    ScaledQuantConvND={1:ScaledQuantConv1D,2:ScaledQuantConv2D,3:ScaledQuantConv3D}
    def __init__(self,dimension,filters,kernel_size,kernel_estimators=3,input_estimators=3,kernel_initializer="random_uniform",
                 input_initializer="random_uniform",
                 kernel_quantizers=quantizers.ShiftedSteSign,
                 input_quantizers=quantizers.ShiftedSteSign,
                 kernel_constraint="weight_clip",
                 kernel_params=None,input_params=None,activation=None,
                 use_bias=False,conv_kwargs=dict(),*args,**kwargs):
        kernel_quantizers,input_quantizers=ABCBase.get_quantizers(kernel_estimators, kernel_quantizers, input_quantizers, kernel_params, input_params)

        estimators=[self.ScaledQuantConvND[dimension](filters,kernel_size,kernel_quantizer=kernel_quantizer,
                                       input_quantizer=input_quantizer,activation=activation,
                                       kernel_constraint=kernel_constraint,
                                       use_bias=use_bias,alpha_trainable=True,**conv_kwargs) 
                    for kernel_quantizer,input_quantizer in zip(kernel_quantizers,input_quantizers)]
        super(ABCConvND,self).__init__(estimators,*args,**kwargs)
        self.kernel_estimators=kernel_estimators
        self.input_estimators=input_estimators
        self.kernel_size=kernel_size
        self.filters=filters
    pass

class ABCConv1D(ABCConvND):
    def __init__(self,*args,**kwargs):
        super(ABCConv1D,self).__init__(1,*args,**kwargs)
        
        
class ABCConv2D(ABCConvND):
    def __init__(self,*args,**kwargs):
        super(ABCConv2D,self).__init__(2,*args,**kwargs)
        
        
class ABCConv3D(ABCConvND):
    def __init__(self,*args,**kwargs):
        super(ABCConv3D,self).__init__(3,*args,**kwargs)
        
class ABCDense(ABCBase):
    def __init__(self,units,kernel_estimators=3,input_estimators=3,kernel_initializer="random_uniform",
                 input_initializer="random_uniform",
                 kernel_quantizers=quantizers.ShiftedSteSign,
                 input_quantizers=quantizers.ShiftedSteSign,
                 kernel_constraint="weight_clip",activation=None,
                 kernel_params=None,
                 input_params=None,
                 use_bias=False,*args,**kwargs):
            
        kernel_quantizers,input_quantizers=ABCBase.get_quantizers(kernel_estimators, kernel_quantizers, input_quantizers, kernel_params, input_params)
        estimators=[ScaledQuantDense(units,kernel_quantizer=kernel_quantizer,
                             input_quantizer=input_quantizer,activation=activation,
                             kernel_constraint=kernel_constraint,
                             use_bias=use_bias,alpha_trainable=True) for kernel_quantizer,input_quantizer in zip(kernel_quantizers,input_quantizers)]
        
        
        super(ABCDense,self).__init__(estimators,*args,**kwargs)
        self.kernel_estimators=kernel_estimators
        self.input_estimators=input_estimators
        self.units=units
    pass