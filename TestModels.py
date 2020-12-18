import Models as M
import Global as G
import numpy as np
import tensorflow as tf
import sys
import time

# PARAMETERS
FUSION = True                   # Type of model: False (mobilenet), True (Fusion)
MODEL = 400                     # Model to be tested
path = "Dataset/test/"          # Dataset:  "Dataset/validation/", "Dataset/train/"

# CREATE AND LOAD THE MODEL
if FUSION :
    model = M.make_mobilenet_fusion( G.IMG_SHAPE )
    model.load_weights( 'models/mobilenet_fusion_weights_' + str(MODEL) )
else :
    model = M.make_mobilenet_model( G.IMG_SHAPE )
    model.load_weights( 'models/mobilenet_weights_' +  str( MODEL ) )

cmd = path + "data.txt"
f = open( cmd, 'r' )
for line in f :
    row = line.split()
    img = tf.io.read_file( path + row[ 0 ] )
    img = tf.image.decode_jpeg( img, channels = 3 )
    img = tf.image.convert_image_dtype( img, tf.float32 )
    img = ( img - 0.5 ) / 0.5
    img = tf.image.resize( img, [ G.IMG_SIZE, G.IMG_SIZE ] )
    img = tf.reshape( img, ( 1, G.IMG_SIZE, G.IMG_SIZE, 3 ) )
    imu = np.array( [ float( row[ 2 ] ), float( row[ 3 ] ), float( row[ 4 ] ),
                      float( row[ 5 ] ), float( row[ 6 ] ), float( row[ 7 ] ) ] )
    imu = np.reshape( imu, ( 1, 6 ) )
    if G.FUSION :
        start_time = time.time()
        a = model.predict( { "img_input": img, "imu_input": imu } )
        time_sec = ( time.time() - start_time )
    else :
        start_time = time.time()
        a = model.predict( img )
        time_sec = ( time.time() - start_time )
    prediction = int( a[ 0 ][ 0 ] >= 0.5 )
    print( prediction, time_sec )
f.close()
