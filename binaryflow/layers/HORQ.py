#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:21:35 2022

@author: ramizouari
"""

import XnorNet

class HORQDense(XnorNet.ScaledQuantDense):
    def __init__(self,degree:int,*args,**kwargs):
        super(HORQDense,self).__init__(*args,**kwargs)
        self.degree=degree
        
    def call(self,inputs,training=False):
        for i in range(self.degree):
            pass
            