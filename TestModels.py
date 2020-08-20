import Models as M
import Global as G
import numpy as np
import tensorflow as tf
import sys
import time

# CREATE AND LOAD THE MODEL
if G.FUSION :
    model = M.make_mobilenet_fusion( G.IMG_SHAPE )
    name = 'models/mobilenet_fusion_weights_%d' % (G.EPOCHS)
    model.load_weights( name )
else :
    model = M.make_mobilenet_model( G.IMG_SHAPE )
    name = 'models/mobilenet_weights_%d' % (G.EPOCHS)
    model.load_weights( name )


#G.FUSION = False
#model = M.make_mobilenet_model( G.IMG_SHAPE )
#name = './../backup_models/07.11.2020/mobilenet_weights_100'
#name = './../backup_models/07.17.2020/mobilenet_weights_550'
#G.FUSION = True
#model = M.make_mobilenet_fusion( G.IMG_SHAPE )
#name = './../backup_models/07.17.2020/mobilenet_fusion_weights_950'
#model.load_weights( name )

#path = "Dataset/test/"
#path = "Dataset/validation/"
path = "Dataset/train/"

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
