#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:32:02 2022

@author: ramizouari
"""

import tensorflow 
import larq
import numpy

class ScaledQuantDense(larq.layers.QuantDense):
    """
        This is a dense binary layer scaled by factors ɑ and β as described by the paper.
    """
    def __init__(self,units,alpha_trainable=False,train_K=False,activation=None,*args,**kwargs):
        self.db_activation=tensorflow.keras.activations.get(activation)
        super(ScaledQuantDense,self).__init__(units,*args,**kwargs)
        self.alpha_trainable=alpha_trainable
        self.train_K=train_K

    def build(self,input_shape):
        super(ScaledQuantDense,self).build(input_shape)
        self.db_dimension=input_shape[-1]
        self.alpha=tensorflow.Variable(tensorflow.divide(tensorflow.norm(self.kernel,axis=0,ord=1),self.db_dimension),
                               trainable=self.alpha_trainable)

    def call(self,inputs,training=False):
        if not self.alpha_trainable and training:
            self.alpha.assign(tensorflow.divide(tensorflow.norm(self.kernel,axis=0,ord=1),self.db_dimension))
        #Result of Quantified dense layer
        Z=super(ScaledQuantDense, self).call(inputs)
        #Calculates the scale factor of the convolution kernel
        #Calculates the scale factor of the input
        beta=tensorflow.divide(tensorflow.norm(inputs,axis=-1,ord=1),self.db_dimension)
        #Calculates the correction tensor
        K=tensorflow.tensordot(beta,self.alpha,axes=0)
        #Apply the correction tensor to the result point-wise
        return self.db_activation(tensorflow.multiply(Z, K))
    
    def call_with_beta(self,inputs,beta,training=False):
        if not self.alpha_trainable and training:
            self.alpha.assign(tensorflow.divide(tensorflow.norm(self.kernel,axis=0,ord=1),self.db_dimension))
        #Result of Quantified dense layer
        Z=super(ScaledQuantDense, self).call(inputs)
        #Calculates the correction tensor
        K=tensorflow.tensordot(beta,self.alpha,axes=0)
        #Apply the correction tensor to the result point-wise
        return self.db_activation(tensorflow.multiply(Z, K))
    
    
    def get_beta(self,inputs):
        return tensorflow.divide(tensorflow.norm(inputs,axis=-1,ord=1),self.db_dimension)
    
    def get_config(self):
        config=super(ScaledQuantDense,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    pass






"""
    This is a 1D convolutional binary layer scaled by factors ɑ and β as described by the paper.
    It will first calculate a Quantified 1D convolution
"""    
    
class ScaledQuantConv1D(larq.layers.QuantConv1D):
    def __init__(self,filters,kernel_size,alpha_trainable=False,*args,**kwargs):
        super(ScaledQuantConv1D,self).__init__(filters,kernel_size,*args,**kwargs)
        self.alpha_trainable=alpha_trainable

    def build(self,input_shape):
        super(ScaledQuantConv1D,self).build(input_shape)
        self.db_dimension=numpy.prod(input_shape[1:])
        self.db_series_span=input_shape[1]
        self.db_series_channels=input_shape[2]
        self.db_ones_tensor=tensorflow.ones(shape=list(self.kernel_size)+[input_shape[-1],1])
        self.alpha=tensorflow.Variable(tensorflow.divide(tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1)),self.db_dimension),self.alpha_trainable)
    def call(self,inputs,training=False):
        if not self.alpha_trainable and training:
            self.alpha.assign(tensorflow.divide(
                tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1)),self.db_dimension))
        #Result of Binarised dense layer
        Z=super(ScaledQuantConv1D, self).call(inputs)
        I=tensorflow.abs(inputs)
        beta=tensorflow.divide(tensorflow.nn.conv1d(
            I,self.db_ones_tensor,self.strides,self.padding.upper()),
            self.db_dimension)
        beta=beta[...,0]
        #Adding scale factors
        K=tensorflow.tensordot(beta,self.alpha,axes=0)
        return tensorflow.multiply(Z, K)
    def get_config(self):
        return super(ScaledQuantConv1D,self).get_config()
    
    
"""
    This is a 2D convolutional binary layer scaled by factors ɑ and β as described by the paper.
    It will first calculate a Quantified 2D convolution
"""


class ScaledQuantConv2D(larq.layers.QuantConv2D):
    def __init__(self,filters,kernel_size,alpha_trainable=False,activation=None,*args,**kwargs):
        super(ScaledQuantConv2D,self).__init__(filters,kernel_size,*args,**kwargs)
        self.alpha_trainable=alpha_trainable
        self.db_activation=tensorflow.keras.activations.get(activation)

    def build(self,input_shape):
        super(ScaledQuantConv2D,self).build(input_shape)
        self.db_img_width=input_shape[1]
        self.db_img_hight=input_shape[2]
        self.db_img_channels=input_shape[3]
        self.db_dimension=numpy.prod(self.kernel_size)*self.db_img_channels
        self.db_ones_tensor=numpy.ones(list(self.kernel_size)+[input_shape[-1],1])
        self.alpha=tensorflow.Variable(tensorflow.divide(
                    tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1,2)),self.db_dimension),
                                   trainable=self.alpha_trainable)

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantConv2D, self).call(inputs)
        #print(f"x:{self.kernel.shape}\t y:{inputs.shape}")
        if training and not self.alpha_trainable:
            self.alpha.assign(tensorflow.divide(
                tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1,2)),self.db_dimension))

        I=tensorflow.abs(inputs)
        beta=tensorflow.divide(tensorflow.nn.conv2d(
            I,self.db_ones_tensor,self.strides,self.padding.upper()),
            self.db_dimension)
        beta=beta[...,0]
        #Adding scale factors
        #print(f"a:{alpha.shape}\t b:{beta.shape}")
        K=tensorflow.tensordot(beta,self.alpha,axes=0)
        R=tensorflow.multiply(Z, K)
        return self.db_activation(R)
    def get_config(self):
        config=super(ScaledQuantConv2D,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    
    
    
class ScaledQuantConv3D(larq.layers.QuantConv3D):
    def __init__(self,*args,**kwargs):
        super(ScaledQuantConv3D,self).__init__(*args,**kwargs)

    def build(self,input_shape):
        super(ScaledQuantConv3D,self).build(input_shape)
        self.db_width=input_shape[1]
        self.db_hight=input_shape[2]
        self.db_depth=input_shape[3]
        self.db_channels=input_shape[4]
        self.db_dimension=numpy.prod(self.kernel_size)*self.db_channels
        self.db_ones_tensor=numpy.ones(list(self.kernel_size)+[input_shape[-1],1])

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantConv3D, self).call(inputs)
        alpha=tensorflow.divide(tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1,2,3)),self.db_dimension)
        I=tensorflow.abs(inputs)
        beta=tensorflow.divide(tensorflow.nn.conv3d(
                I,self.db_ones_tensor,self.strides,self.padding.upper()),
                self.db_dimension)
        beta=beta[...,0]
        #Adding scale factors
        K=tensorflow.tensordot(beta,alpha,axes=0)
        return tensorflow.multiply(Z, K)
    def get_config(self):
        return super(ScaledQuantConv3D,self).get_config()


XnorDense=ScaledQuantDense
XnorConv1D=ScaledQuantConv1D
XnorConv2D=ScaledQuantConv2D
XnorConv3D=ScaledQuantConv3D



class ScaledQuantPlusDense(larq.layers.QuantDense):
    """
        This is a dense binary layer scaled by factors ɑ and β as described by the paper.
    """
    def __init__(self,units,scale_initializer="he_normal",activation=None,*args,**kwargs):
        self.db_activation=tensorflow.keras.activations.get(activation)
        super(ScaledQuantPlusDense,self).__init__(units,*args,**kwargs)
        self.scale_initializer=tensorflow.keras.initializers.get(scale_initializer)

    def build(self,input_shape):
        super(ScaledQuantPlusDense,self).build(input_shape)
        self.db_dimension=input_shape[-1]
        self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.units,)))
        
    def call(self,inputs,training=False):
       #Result of Quantified dense layer
        Z=super(ScaledQuantPlusDense, self).call(inputs)
        K=self.alpha
        #Apply the correction tensor to the result point-wise
        return self.db_activation(tensorflow.multiply(Z, K))
    
    def get_config(self):
        config=super(ScaledQuantPlusDense,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    pass



class ScaledQuantPlusConv2D(larq.layers.QuantConv2D):
    def __init__(self,filters,kernel_size,mode=1,scale_initializer="he_normal",activation=None,*args,**kwargs):
        super(ScaledQuantPlusConv2D,self).__init__(filters,kernel_size,*args,**kwargs)
        self.db_activation=tensorflow.keras.activations.get(activation)
        self.mode=mode
        self.scale_initializer=tensorflow.keras.initializers.get(scale_initializer)

    def build(self,input_shape):
        super(ScaledQuantPlusConv2D,self).build(input_shape)
        output_shape=self.compute_output_shape(input_shape)
        if self.mode==1:
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.filters,)))
        elif self.mode==2:
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=output_shape[1:]))
        elif self.mode==3:
            self.beta=tensorflow.Variable(self.scale_initializer(shape=output_shape[1:3]))
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.filters,)))
        elif self.mode==4:
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.filters,)))
            self.beta=[tensorflow.Variable(self.scale_initializer(shape=(dim,))) for dim in output_shape[1:3]]
        else:
            raise RuntimeError(f"Unknown mode {self.mode}")

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantPlusConv2D, self).call(inputs)
        if self.mode<=2:
            K=self.alpha
        elif self.mode==3:
            K=tensorflow.tensordot(self.beta,self.alpha,axes=0)
        elif self.mode==4:
            K=tensorflow.tensordot(tensorflow.tensordot(self.beta[0], self.beta[1], axes=0),
                                   self.alpha,axes=0)
        R=tensorflow.multiply(Z, K)
        return self.db_activation(R)
    def get_config(self):
        config=super(ScaledQuantPlusConv2D,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    
class ScaledQuantPlusConv1D(larq.layers.QuantConv1D):
    def __init__(self,filters,kernel_size,mode=1,scale_initializer="he_normal",activation=None,*args,**kwargs):
        super(ScaledQuantPlusConv1D,self).__init__(filters,kernel_size,*args,**kwargs)
        self.db_activation=tensorflow.keras.activations.get(activation)
        self.mode=mode
        self.scale_initializer=tensorflow.keras.initializers.get(scale_initializer)

    def build(self,input_shape):
        super(ScaledQuantPlusConv1D,self).build(input_shape)
        output_shape=self.compute_output_shape(input_shape)
        if self.mode==1:
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.filters,)))
        elif self.mode==2:
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=output_shape[1:]))
        elif self.mode==3:
            self.beta=tensorflow.Variable(self.scale_initializer(shape=output_shape[1]))
            self.alpha=tensorflow.Variable(self.scale_initializer(shape=(self.filters,)))
        else:
            raise RuntimeError(f"Unknown mode {self.mode}")

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantPlusConv1D, self).call(inputs)
        if self.mode<=2:
            K=self.alpha
        elif self.mode==3:
            K=tensorflow.tensordot(self.beta,self.alpha,axes=0)
        R=tensorflow.multiply(Z, K)
        return self.db_activation(R)
    def get_config(self):
        config=super(ScaledQuantPlusConv1D,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    
XnorPlusDense=ScaledQuantPlusDense
XnorPlusConv2D=ScaledQuantPlusConv2D

#%%
