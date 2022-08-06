#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 22:16:36 2022

@author: ramizouari
"""

from tensorflow.keras.layers import Layer


class ImageNormalizationLayer(Layer):
    """
    This is a simple normalisation layer for image data
    It will divide the pixel intensity of each channel by the max intensity
    """
    def __init__(self,max_intensity=255.):
        super(ImageNormalizationLayer, self).__init__()
        self.max_intensity=max_intensity
        
    def call(self,I):
        return I/self.max_intensity
    def get_config(self):
        return {"max_intensity": self.max_intensity}




