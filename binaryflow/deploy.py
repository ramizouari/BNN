#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:03:54 2022

@author: ramizouari
"""
import tensorflow as tf
import numpy as np


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
    m1=t(0x5555555555555555)
    m2=t(0x3333333333333333)
    m3=t(0x0F0F0F0F0F0F0F0F)
    m4=t(0x0101010101010101)
    s55 = t(m1 & mask)  # Add more digits for 128bit support
    s33 = t(m2 & mask)
    s0F = t(m3 & mask)
    s01 = t(m4 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))

def bit_count_tensor(arr):
    # Make the values type-agnostic (as long as it's integers)
    mask = (-1)
    m1=(0x5555555555555555)
    m2=(0x3333333333333333)
    m3=(0x0F0F0F0F0F0F0F0F)
    m4=(0x0101010101010101)
    s55 = (m1 & mask)  # Add more digits for 128bit support
    s33 = (m2 & mask)
    s0F = (m3 & mask)
    s01 = (m4 & mask)

    arr = arr - tf.bitwise.bitwise_and(tf.bitwise.right_shift(arr, 1), s55)
    arr = tf.bitwise.bitwise_and(arr, s33) + tf.bitwise.bitwise_and(tf.bitwise.right_shift(arr, 2), s33)
    arr = tf.bitwise.bitwise_and(arr + tf.bitwise.right_shift(arr, 4), s0F)
    return tf.bitwise.right_shift(arr * s01, 8 * (arr.dtype.size - 1))


def binarize(u,retPaddingCount=False,r1=None,r2=None):
    #Calculates the padding parameters r1 and r2
    if r1 is None and r2 is None:
        #r1 is the padding added so that the function packbits properly works
        #Divide by 8, then take  modulo 8. Thanks to the binary representation of integers, this is equivalent to a right shift by 3, than a bitwise-and with 7
        r1=(-u.shape[-1]>>3)&0x7
        #r2 is the padding added so that transforming the array to 64 representation works.
        #Take modulo 8.Thank to the binary representation of integers, this is equivalent to a bitwise-and with 7
        r2=(-u.shape[-1])&0x7
        #The total added padding 
    r=(r1<<3)+r2    
    #If the array has a rank superior to one, apply recursively the binarize function along its first axis
    if len(u.shape) > 1:
        X=np.array([binarize(s,r1=r1,r2=r2) for s in u])
        return X if not retPaddingCount else (X,r)
    
    #Transform the array 
    v=np.packbits(np.pad(u==1,(0,r2)))
    R=np.pad(v,(0,r1)).view(np.uint64)
    return R if not retPaddingCount else (R,r)

def binarize_tensor(u,retPaddingCount=False,r1=None,r2=None):
    #Calculates the padding parameters r1 and r2
    if r1 is None and r2 is None:
        #r1 is the padding added so that the function packbits properly works
        #Divide by 8, then take  modulo 8. Thanks to the binary representation of integers, this is equivalent to a right shift by 3, than a bitwise-and with 7
        r1=(-u.shape[-1]>>3)&0x7
        #r2 is the padding added so that transforming the array to 64 representation works.
        #Take modulo 8.Thank to the binary representation of integers, this is equivalent to a bitwise-and with 7
        r2=(-u.shape[-1])&0x7
        #The total added padding 
    r=(r1<<3)+r2    
    #If the array has a rank superior to one, apply recursively the binarize function along its first axis
    if len(u.shape) > 1:
        X=tf.map_fn(lambda s:binarize(s,r1=r1,r2=r2),u,dtype=tf.uint64)
        return X if not retPaddingCount else (X,r)
    
    #Transform the array 
    v=np.packbits(np.pad(u==1,(0,r2)))
    R=np.pad(v,(0,r1)).view(np.uint64)
    return R if not retPaddingCount else (R,r)


def unbinarize(u,r=0,_rec=False):
    if not _rec and len(u.shape)==1:
        v=np.unpackbits(u.view(np.uint8))
        v=v[:v.shape[-1]-r]
        return (np.left_shift(v.astype(np.int8),1)-1).astype(float)
    elif len(u.shape) > 1:
        X=np.array([unbinarize(s,_rec=True) for s in u])
        return X[...,:X.shape[-1]-r]
    else:
        v=np.unpackbits(u.view(np.uint8))
        return (np.left_shift(v.astype(np.int8),1)-1).astype(float)
                           
def binarized_mat_mul(A,u,padding=None):
    if not padding:
        u,r=binarize(u,retPaddingCount=True)
    n=A.shape[-1]
    m=A.shape[0]
    x=np.sum(bit_count(~(A^u)),axis=-1)
    return (x.astype(np.int32)<<1) - (n<<6) - r
    
def binarized_batch_mat_mul(A,U,padding=None):
    if not padding:
        U,r=binarize(U.T,retPaddingCount=True)
    n=A.shape[-1]
    m=A.shape[0]
    #return np.apply_along_axis(BinarizedScalar,-1,A)
    X=np.apply_along_axis(lambda u:np.sum(bit_count(~(A^u)),axis=-1),-1,U)
    return (X.astype(np.int32)<<1) - (n<<6) - r
    
def get_binarized_shape(U=None,shape=None):
    
    if U is not None:
        return type(U.shape)([*U.shape[:-1],(U.shape[-1]+63)//64])
    if shape is not None:
        return type(shape)([*shape[:-1],(shape[-1]+63)//64])    
    

def binarized_batch_mat_mul_tensor(A,U,padding=None,transpose=True):
    if transpose:
        U=tf.transpose(U)
    if not padding:
        U,r=tf.numpy_function(lambda X:binarize(X,retPaddingCount=True),[U],(tf.uint64,tf.int64))
    n=A.shape[-1]
    m=A.shape[0]
    #return np.apply_along_axis(BinarizedScalar,-1,A)
    X=tf.map_fn(lambda u:tf.reduce_sum(bit_count_tensor(tf.bitwise.invert(tf.bitwise.bitwise_xor(A,u))),axis=-1),U)
    z=tf.bitwise.left_shift(tf.cast(X,tf.int64),1)
    y=tf.constant((n<<6),dtype=tf.int64)+ tf.cast(r,tf.int64)
    return  z - y
"""
def BinarizedMatConv2D(K,I,padding=None):
    if not padding:
        I,r=binarize(I,retPaddingCount=True)
    w,h=K.shape[1:3]
    n,m,C_in=I.shape[1:]
    C_out=K.shape[-1]
    dim=n*m*C_out
    #J=np.zeros(I.shape[])
    for i in range(n): 
        for j in range(m):
            for c in range(C):
                u=np.pad(I[max(i-n//2,0):min(i+n//2+1,n-1)][max(j-m//2,j+m//2+m-1)],[0,(max(0,n//2-i),max(n//2-n+1+i,0)),(max(0,m//2-j),max(m//2-m+1+j,0)),0]).flatten()
                v=K.flatten()

                x=np.sum(bit_count(~(u^v)))
                J[i,j,c]=(x.astype(np.int32)<<1) - (dim<<6)
   """
   