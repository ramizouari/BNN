import tensorflow as tf
from tensorflow import keras

import load
import larq as lq
import numpy as np
from tensorflow.keras import optimizers,losses

import layers


data_format="channels_last"


(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
quantization=dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)
y_train=tf.one_hot(y_train, 10)
y_test=tf.one_hot(y_test, 10)
X_train=X_train.astype(dtype=np.float32)
X_test=X_test.reshape(list(X_test.shape)+[1])
X_train=X_train.reshape(list(X_train.shape)+[1])


cnn_model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(192,kernel_size=(5,5),input_shape=(32,32,1),
                           activation="relu",padding="same", data_format=data_format),
    tf.keras.layers.Conv2D(64,kernel_size=(1,1),activation="relu",padding="same", data_format=data_format),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=2, data_format=data_format),
   # lq.layers.QuantConv2D(192, kernel_size=(1,1), input_quantizer="ste_sign",
    #    kernel_quantizer="ste_sign",
     #   kernel_constraint="weight_clip", data_format='channels_first'),
    #lq.layers.QuantConv2D(64, kernel_size=(1,1), input_quantizer="ste_sign",
     #   kernel_quantizer="ste_sign",
      #  kernel_constraint="weight_clip", data_format='channels_first'),
      tf.keras.layers.Conv2D(192,kernel_size=(5,5),activation="relu",padding="same", data_format=data_format),
    tf.keras.layers.Conv2D(96,kernel_size=(1,1),activation="relu",padding="same", data_format=data_format),

    tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=2, data_format=data_format),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
    
    ])


Conv2D=layers.ScaledQuantConv2D
Dense=layers.ScaledQuantDense


bnn_model=tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(stddev=4,input_shape=(28,28,1)),
    layers.ImageNormalisationLayer(),
    tf.keras.layers.RandomCrop(width=28,height=28),
    tf.keras.layers.Conv2D(192, 3,
                          use_bias=False),
    tf.keras.layers.ReLU(),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    Conv2D(160, 1, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.ReLU(),
    Conv2D(92, 1, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),
    

    Conv2D(192, 5, padding="same",data_format=data_format,**quantization),
#    tf.keras.layers.Dropout(0.3),
    Conv2D(160, 1, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.ReLU(),
    tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(2,2)),

    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    Conv2D(72, 3, padding="same",**quantization),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format=data_format),

    Conv2D(72, 3, padding="same",data_format=data_format,**quantization),
    Conv2D(72, 1, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),
    tf.keras.layers.ReLU(),


    #tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    Dense(128,**quantization),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    Dense(128,**quantization),

    tf.keras.layers.Dense(10,activation="softmax")
    
    ])

optimiser=optimizers.Adam(learning_rate=0.01,decay=0.00001)
loss=losses.CategoricalCrossentropy(from_logits=False)
bnn_model.compile(optimizer=optimiser,loss=loss,metrics=["accuracy"])
bnn_model.fit(X_train,y_train,epochs=40,batch_size=128,validation_data=(X_test,y_test))