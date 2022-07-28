import tensorflow
import larq
import numpy
import quantizers

"""
    Convention: To avoid any unintentional naming conflict, we will add the prefix "db_" to every 
    member variable name.
"""



"""
This is a simple normalisation layer for image data
It will divide the pixel intensity of each channel by the max intensity
"""
class ImageNormalisationLayer(tensorflow.keras.layers.Layer):
    def __init__(self,max_intensity=255):
        super(ImageNormalisationLayer, self).__init__()
        self.max_intensity=max_intensity
        
    def call(self,I):
        return I/self.max_intensity
    def get_config(self):
        return {"max_intensity": self.max_intensity}


"""
    This is a dense binary layer scaled by factors ɑ and β as described by the paper.
"""

class ScaledQuantDense(larq.layers.QuantDense):
    def __init__(self,units,alpha_trainable=False,train_K=False,*args,**kwargs):
        if "activation" in kwargs:
            activation=kwargs["activation"]
            self.db_activation=activation
        else:
            self.db_activation=None
        super(ScaledQuantDense,self).__init__(units,*args,**{k:kwargs[k] for k in kwargs if k!="activation"})
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
        return tensorflow.multiply(Z, K)
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
    def __init__(self,filters,kernel_size,alpha_trainable=False,*args,**kwargs):
        super(ScaledQuantConv2D,self).__init__(filters,kernel_size,*args,**kwargs)
        self.alpha_trainable=alpha_trainable

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
        beta=beta[:,:,0]
        #Adding scale factors
        #print(f"a:{alpha.shape}\t b:{beta.shape}")
        K=tensorflow.tensordot(beta,alpha,axes=0)
        R=tensorflow.multiply(Z, K)
        return R if not self.db_activation else self.db_activation(R)
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
        beta=beta[:,0]
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
        beta=beta[:,:,:,0]
        #Adding scale factors
        K=tensorflow.tensordot(beta,alpha,axes=0)
        return tensorflow.multiply(Z, K)
    def get_config(self):
        return super(ScaledQuantConv3D,self).get_config()



class ABCDense(tensorflow.keras.layers.Layer):
    def __init__(self,units,kernel_estimators=3,activation_estimators=3,kernel_initializer="random_uniform",
                 activation_initialize="random_uniform",
                 kernel_quantizer=quantizers.ShiftedSteSign(),
                 input_quantizer=quantizers.ShiftedSteSign(),
                 kernel_constraint="weight_clip",activation=None,
                 use_bias=False,*args,**kwargs):
        super(ABCDense,self).__init__(*args,**kwargs)
        self.kernel_estimators=kernel_estimators
        self.activation_estimators=activation_estimators
        self.units=units
        self.estimators=[ScaledQuantDense(units,kernel_quantizer=kernel_quantizer,
                                       input_quantizer=input_quantizer,activation=activation,
                                       kernel_constraint=kernel_constraint,
                                       use_bias=use_bias,alpha_trainable=True) for i in range(kernel_estimators)]
    pass
    
    def build(self,input_shape):
        #self.kernels=tensorflow.Variable()
        super(ABCDense,self).build(input_shape)
        for estimator in self.estimators:
            estimator.build(input_shape)
        pass
    
    def call(self,inputs,training=False):
        output=0
        for estimator in self.estimators:
            output+=estimator.call(inputs,training)
        return output
            


class ABCConv2D(tensorflow.keras.layers.Layer):
    pass

