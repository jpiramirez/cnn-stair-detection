import Models as M
import Global as G
import numpy as np
import tensorflow as tf
import sys
import time

# CREATE AND LOAD THE MODEL
if G.FUSION :
    model = M.make_mobilenet_fusion_3class( G.IMG_SHAPE )
    name = 'models/mobilenet_fusion_3class_weights_%d' % (G.EPOCHS)
    #name = './../backup_models/08.21.2020/mobilenet_fusion_3class_weights_%d' % (G.EPOCHS)
    model.load_weights( name )
else :
    model = M.make_mobilenet_model_3class( G.IMG_SHAPE )
    name = 'models/mobilenet_3class_weights_%d' % (G.EPOCHS)
    #name = './../backup_models/08.21.2020/mobilenet_3class_weights_%d' % (G.EPOCHS)
    model.load_weights( name )

path = "Dataset/test/"

cmd = path + "data_3class_new.txt"  #"data_3class.txt"
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
    print( np.argmax( a ), time_sec )

f.close()
