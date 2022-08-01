#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 2022

@author: ramizouari
"""

import tensorflow
import larq
import numpy
import quantizers
from collections.abc import Iterable

"""
    Convention: To avoid any unintentional naming conflict, we will add the prefix "db_" to every 
    member variable name.
"""




class ImageNormalisationLayer(tensorflow.keras.layers.Layer):
    """
    This is a simple normalisation layer for image data
    It will divide the pixel intensity of each channel by the max intensity
    """
    def __init__(self,max_intensity=255.):
        super(ImageNormalisationLayer, self).__init__()
        self.max_intensity=max_intensity
        
    def call(self,I):
        return I/self.max_intensity
    def get_config(self):
        return {"max_intensity": self.max_intensity}




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
        if self.alpha_trainable:
            self.alpha=tensorflow.Variable(tensorflow.divide(tensorflow.norm(self.kernel,axis=0,ord=1),self.db_dimension),
                               trainable=True)

    def call(self,inputs,training=False):
        #Result of Quantified dense layer
        Z=super(ScaledQuantDense, self).call(inputs)
        #Calculates the scale factor of the convolution kernel
        alpha= self.alpha if self.alpha_trainable else \
            tensorflow.divide(tensorflow.norm(self.kernel,axis=0,ord=1),self.db_dimension)
        
        #Calculates the scale factor of the input
        beta=tensorflow.divide(tensorflow.norm(inputs,axis=-1,ord=1),self.db_dimension)
        #Calculates the correction tensor
        K=tensorflow.tensordot(beta,alpha,axes=0)
        #Apply the correction tensor to the result point-wise
        return self.db_activation(tensorflow.multiply(Z, K))
    def get_config(self):
        config=super(ScaledQuantDense,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config
    pass

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
        if self.alpha_trainable:
            self.alpha=tensorflow.Variable(tensorflow.divide(
                    tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1,2)),self.db_dimension),
                                   trainable=True)

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantConv2D, self).call(inputs)
        #print(f"x:{self.kernel.shape}\t y:{inputs.shape}")
        alpha=self.alpha if self.alpha_trainable else \
            tensorflow.divide(
                tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1,2)),self.db_dimension)
        I=tensorflow.abs(inputs)
        beta=tensorflow.divide(tensorflow.nn.conv2d(
            I,self.db_ones_tensor,self.strides,self.padding.upper()),
            self.db_dimension)
        beta=beta[...,0]
        #Adding scale factors
        #print(f"a:{alpha.shape}\t b:{beta.shape}")
        K=tensorflow.tensordot(beta,alpha,axes=0)
        R=tensorflow.multiply(Z, K)
        return self.db_activation(R)
    def get_config(self):
        config=super(ScaledQuantConv2D,self).get_config()
        config.update({"alpha_trainable":self.alpha_trainable})
        return config

"""
    This is a 1D convolutional binary layer scaled by factors ɑ and β as described by the paper.
    It will first calculate a Quantified 1D convolution
"""    
    
class ScaledQuantConv1D(larq.layers.QuantConv1D):
    def __init__(self,*args,**kwargs):
        super(ScaledQuantConv1D,self).__init__(*args,**kwargs)

    def build(self,input_shape):
        super(ScaledQuantConv1D,self).build(input_shape)
        self.db_dimension=numpy.prod(input_shape[1:])
        self.db_series_span=input_shape[1]
        self.db_series_channels=input_shape[2]
        self.db_ones_tensor=numpy.ones(list(self.kernel_size)+[input_shape[-1],1])

    def call(self,inputs,training=False):
        #Result of Binarised dense layer
        Z=super(ScaledQuantConv1D, self).call(inputs)
        alpha=tensorflow.divide(
            tensorflow.reduce_sum(tensorflow.abs(self.kernel),axis=(0,1)),self.db_dimension)
        I=tensorflow.abs(inputs)
        beta=tensorflow.divide(tensorflow.nn.conv1d(
            I,self.db_ones_tensor,self.strides,self.padding.upper()),
            self.db_dimension)
        beta=beta[...,0]
        #Adding scale factors
        K=tensorflow.tensordot(beta,alpha,axes=0)
        return tensorflow.multiply(Z, K)
    def get_config(self):
        return super(ScaledQuantConv1D,self).get_config()
    
    
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





class ABCBase(tensorflow.keras.layers.Layer):
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
        super(ABCBase,self).__init__(*args,**kwargs)
        self.kernel_estimators=len(estimators)
        self.estimators=estimators
    pass
    
    def build(self,input_shape):
        #self.kernels=tensorflow.Variable()
        super(ABCBase,self).build(input_shape)
        for estimator in self.estimators:
            estimator.build(input_shape)
        pass
    
    def call(self,inputs,training=False):
        output=0
        for estimator in self.estimators:
            output+=estimator.call(inputs,training)
        return output
    
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


"""ww
    ABCDense is a class 
"""

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
        
        
class BiRealBaseBlock(tensorflow.keras.layers.Layer):
    def __init__(self,estimator_type,*args,**kwargs):
        super(BiRealBaseBlock,self).__init__()
        self.batchNorm=tensorflow.keras.layers.BatchNormalization()
        self.estimator_type=estimator_type
        self.estimator_metadata={"args":args,"kwargs":kwargs}
    
    def build(self,input_shape):
        super(BiRealBaseBlock,self).build(input_shape)
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[1:],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        
        pass
    
    def build_bach_norm(self,*args,**kwargs):
        self.batchNorm=tensorflow.keras.layers.BatchNormalization(*args,**kwargs)
        return self
    
    def call(self,inputs,training=False):
        super(BiRealBaseBlock,self).call(inputs)
        return self.estimator(self.batchNorm(inputs))+inputs
        pass
    
    
class BiRealDenseBlock(BiRealBaseBlock):
    def __init__(self,estimator_type=larq.layers.QuantDense,*args,**kwargs):
        self.estimator_type=estimator_type
        super(BiRealDenseBlock, self).__init__(estimator_type,*args,**kwargs)
    
    def build(self,input_shape):
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[-1],*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        pass
    
    
class BiRealConv2DBlock(BiRealBaseBlock):
    def __init__(self,kernel,estimator_type=larq.layers.QuantConv2D,*args,**kwargs):
        self.estimator_type=estimator_type
        super(BiRealConv2DBlock, self).__init__(estimator_type,*args,**kwargs)
        self.kernel_shape=kernel
    
    def build(self,input_shape):
        self.batchNorm.build(input_shape)
        self.estimator=self.estimator_type(input_shape[-1],self.kernel_shape,*self.estimator_metadata["args"],
                                           **self.estimator_metadata["kwargs"])
        pass
     
#%%
