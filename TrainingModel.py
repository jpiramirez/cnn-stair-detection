import tensorflow as tf
import Models as M
import Global as G
#import pathlib
import os
import numpy as np
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

# LOAD DATA (TEXT MODE)
def loadData( path ) :
    cmd = path + "data.txt"
    f = open( cmd, 'r' )
    names = [] # Create a list with the image names
    label = [] # Create a list with the labels
    for line in f :
        row = line.split()
        names.append( path + str( row[ 0 ] ) ) # Name of the image from the current folder
        label.append( int( row[ 1 ] ) )
    f.close()
    full_list = list( zip( names, label ) )
    random.shuffle( full_list )
    names, label = zip( *full_list )
    names = list( names )
    label = list( label )
    names_tensor = tf.convert_to_tensor( names, dtype = tf.string )
    label_tensor = tf.convert_to_tensor( label, dtype = tf.int32 )
    return names_tensor, label_tensor

train_names, train_label = loadData( "Dataset/train/" )
val_names, val_label = loadData( "Dataset/validation/" )

# GET DATA GIVEN A FILE PATH
def get_image( file_path ) :
    img = tf.io.read_file( file_path )                                          # Read an image from a path (here the path is a tensor)
    img = tf.image.decode_jpeg( img, channels = 3 )
    img = tf.image.convert_image_dtype( img, tf.float32 )                       # To convert to floats in the [0,1] range.
    img = ( img - 0.5 ) / 0.5                                                     # To convert the image in the [-1,1] range.
    img = tf.image.resize(img, [G.IMG_SIZE, G.IMG_SIZE])
    return img

def process_data_train( file_path ) :
    img = get_image( file_path )
    a = tf.strings.regex_full_match( train_names, file_path )                   # Search in the name column
    index = tf.where( a )                                                       # Find the index
    index = tf.reshape( index, () )                                             # Flat the tensor
    return img, train_label[ index ]

def process_data_val( file_path ) :
    img = get_image( file_path )
    a = tf.strings.regex_full_match( val_names, file_path )                   # Search in the name column
    index = tf.where( a )                                                       # Find the index
    index = tf.reshape( index, () )                                             # Flat the tensor
    return img, val_label[ index ]

def prepare_dataset( ds, shuffle_buffer_size = 1000, batch_size = 24 ) :
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)                            # Buffer size (element to be considered in the shuffle, True shuffle buffer_size >= data_size)
    ds = ds.batch( batch_size )
    ds = ds.prefetch( buffer_size=AUTOTUNE )                                    # Lets the dataset fetch batches in the background while the model is training.
    return ds

# CREATE THE DATASET
train_dataset = tf.data.Dataset.from_tensor_slices( train_names )
train_labeled_ds = train_dataset.map( process_data_train, num_parallel_calls = AUTOTUNE )
train_ds = prepare_dataset( train_labeled_ds,
                           shuffle_buffer_size = len(train_names),
                           batch_size = G.BATCH_SIZE_TRAINING )

val_dataset = tf.data.Dataset.from_tensor_slices( val_names )
val_labeled_ds = val_dataset.map( process_data_val, num_parallel_calls = AUTOTUNE )
val_ds = prepare_dataset( val_labeled_ds,
                           shuffle_buffer_size = len(val_names),
                           batch_size = G.BATCH_SIZE_VALIDATION )

# CREATE THE MODEL
model = M.make_mobilenet_model( G.IMG_SHAPE )

# TRAINING THE MODEL
model.fit(
    train_ds,
    epochs = G.EPOCHS,
    validation_data = val_ds,
    )

# SAVE THE MODEL
path = 'models/'
cmd = 'mkdir ' + path
os.system( cmd )
filename = 'mobilenet_weights_%d' % (G.EPOCHS)
model.save_weights( path + filename )
