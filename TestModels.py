import Models as M
import numpy as np
import tensorflow as tf
import sys
import time

# HELP DISPLAY
if len( sys.argv ) == 2:
    if sys.argv[ 1 ] == '--help':
        print('\n Mobilenet training\n')
        print(' This command: python3 TestModels.py --help')
        print(' Template:     python3 TestModels.py --options values > labels.txt');
        print(' Example:      python3 TestModels.py --dataset Dataset/ > labels.txt')
        print('\n List of options')
        print('    --dataset  : <string>  Folder with the dataset, default: Dataset/test');
        print('    --imgsize  : <integer> Image size, default: 224')
        print('    --modeldir : <string>  Directory with the models, default: models');
        print('    --modeltype: <string>  Type: {mobilenet, fusion}, default: mobilenet');
        print('    --modelid  : <integer> Model id, default: 400')
        sys.exit( 1 )

# DEFAULT PARAMETERS
DATASET = 'Dataset/test'
IMG_SIZE = 224
MODEL_DIR = 'models'
MODEL_TYPE = 'mobilenet'
MODEL_ID = 400

# LOAD PARAMETER VALUES
i = 1
while i < len( sys.argv ) :
    cmd = str( sys.argv[ i ] )
    i = i + 1
    if cmd == '--dataset' :
        DATASET = str( sys.argv[ i ] )
    elif cmd == '--imgsize' :
        IMG_SIZE = int( sys.argv[ i ] )
    elif cmd == '--modeldir' :
        MODEL_DIR = str( sys.argv[ i ] )
    elif cmd == "--modeltype" :
        MODEL_TYPE = str( sys.argv[ i ] )
    elif cmd == '--modelid' :
        MODEL_ID = int( sys.argv[ i ] )
    i = i + 1
    
# CREATE AND LOAD THE MODEL
if MODEL_TYPE == 'fusion':
    model = M.make_mobilenet_fusion( (IMG_SIZE, IMG_SIZE, 3) )
    model.load_weights( MODEL_DIR + '/mobilenet_fusion_weights_' + str(MODEL_ID) )
else :
    model = M.make_mobilenet_model( (IMG_SIZE, IMG_SIZE, 3) )
    model.load_weights( MODEL_DIR + '/mobilenet_weights_' +  str( MODEL_ID ) )

cmd = DATASET + "/data.txt"
f = open( cmd, 'r' )
for line in f :
    row = line.split()
    img = tf.io.read_file( DATASET + "/" + row[ 0 ] )
    img = tf.image.decode_jpeg( img, channels = 3 )
    img = tf.image.convert_image_dtype( img, tf.float32 )
    img = ( img - 0.5 ) / 0.5
    img = tf.image.resize( img, [ IMG_SIZE, IMG_SIZE ] )
    img = tf.reshape( img, ( 1, IMG_SIZE, IMG_SIZE, 3 ) )
    imu = np.array( [ float( row[ 2 ] ), float( row[ 3 ] ), float( row[ 4 ] ),
                      float( row[ 5 ] ), float( row[ 6 ] ), float( row[ 7 ] ) ] )
    imu = np.reshape( imu, ( 1, 6 ) )
    if MODEL_TYPE == 'fusion':
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
