import tensorflow as tf
from tensorflow import keras

import load
import larq as lq

from tensorflow.keras import optimizers,losses

import layers


data_format="channels_last"

X_train,y_train=load.load_cifar("dataset/cifar-10","data*",data_format=data_format)
y_train=tf.one_hot(y_train,10)
X_test,y_test=load.load_cifar("dataset/cifar-10","test_batch",data_format=data_format)

quantization=dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)


cnn_model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(192,kernel_size=(5,5),activation="relu",padding="same", data_format=data_format),
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

bnn_model=tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(stddev=4),
    layers.ImageNormalisationLayer(),
    tf.keras.layers.RandomCrop(width=28,height=28),
    tf.keras.layers.Conv2D(192, 3,
                          use_bias=False,
                          input_shape=(3, 32, 32)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    lq.layers.QuantConv2D(160, 1, padding="valid",data_format=data_format,**quantization),
    lq.layers.QuantConv2D(92, 1, padding="valid",data_format=data_format,**quantization),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),
    

    lq.layers.QuantConv2D(192, 5, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.Dropout(0.3),
    lq.layers.QuantConv2D(160, 1, padding="valid",data_format=data_format,**quantization),
    tf.keras.layers.ReLU(),
    tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=(2,2)),

    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    lq.layers.QuantConv2D(72, 3, padding="same",**quantization),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format=data_format),

    lq.layers.QuantConv2D(72, 3, padding="same",data_format=data_format,**quantization),
    lq.layers.QuantConv2D(72, 1, padding="same",data_format=data_format,**quantization),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),
    tf.keras.layers.ReLU(),


    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),

    lq.layers.QuantDense(128,**quantization),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    lq.layers.QuantDense(128,**quantization),
    tf.keras.layers.BatchNormalization(momentum=0.1, scale=False),

    tf.keras.layers.Dense(10,activation="softmax")
    
    ])

optimiser=optimizers.Adam(learning_rate=0.01,decay=0.00001)
loss=losses.CategoricalCrossentropy(from_logits=False)
bnn_model.compile(optimizer=optimiser,loss=loss,metrics=["accuracy"])
bnn_model.fit(X_train,y_train,epochs=40,batch_size=128,validation_data=(X_test,tf.one_hot(y_test,10)))