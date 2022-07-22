import tensorflow as tf
import larq

"""
This is a simple normalisation layer for image data
"""
class ImageNormalisationLayer(tf.keras.layers.Layer):
    def __init__(self,max_intensity=255):
        super(ImageNormalisationLayer, self).__init__()
        self.max_intensity=max_intensity
        
    def call(self,I):
        return I/self.max_intensity
    




"""
    This is a 2D convolutional binary layer scaled by factors ɑ and β as described by the paper
"""

class ScaledBinaryConv2D(larq.layers.QuantConv2D):
    def __init__(self):
        super(WeightedBinaryLayer,self).__init__(self)
        
    
    def call(self,I):
        pass
    

"""
    This is a dense binary layer scaled by factors ɑ and β as described by the paper
"""
class ScaledDenseLayer(larq.layers.QuantDense):
    pass




