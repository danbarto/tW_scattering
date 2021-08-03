## Define all the different models here


import tensorflow as tf
from keras.utils import np_utils

def baseline_model(input_dim, out_dim):
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5*input_dim, input_dim=input_dim, activation='relu'))
    model.add( tf.keras.layers.Dropout( rate = 0.1 ) )
    model.add(tf.keras.layers.Dense(5*input_dim, activation='relu')) 
    model.add( tf.keras.layers.Dropout( rate = 0.1 ) ) # this introduces some stochastic behavior

    model.add(tf.keras.layers.Dense(out_dim, activation='softmax'))
    
    ## From initial studies, RMSprop with learning rate of 0.001 performs best.
    ## We can reiterate this, different optimizers are SGD, Adam
    ## Examples: 
    # tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.RMSprop(lr=0.001) ## performs best.

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model

def normalization_model(input_dim, out_dim):
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5*input_dim, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization() )
    model.add( tf.keras.layers.Dropout( rate = 0.1 ) )
    model.add(tf.keras.layers.Dense(5*input_dim, activation='relu')) 
    model.add(tf.keras.layers.BatchNormalization() )
    model.add( tf.keras.layers.Dropout( rate = 0.1 ) ) # this introduces some stochastic behavior

    model.add(tf.keras.layers.Dense(out_dim, activation='softmax'))
    
    opt = tf.keras.optimizers.RMSprop(lr=0.001) ## performs best.

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model
