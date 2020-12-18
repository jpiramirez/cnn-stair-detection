import cv2
import os
import sys

# HELP DISPLAY
if len( sys.argv ) == 2:
    if sys.argv[ 1 ] == '--help':
        print('\n Obstacle detection \n')
        print(' This command: python3 ObstacleDection.py --help')
        print(' Template:     python3 ObstacleDection.py --options values > labels.txt');
        print(' Example:      python3 ObstacleDection.py --dataset Dataset/test > labels')
        print('\n List of options')
        print('    --dataset: <string>  Folder with the dataset, default: Dataset/test');
        print('    --predict: <string>  Labels with the prediction, default: result/mobilenet_400_test.txt');
        print('    --odir   : <string>  Labels with the prediction, default: result/test_mobilenet_400');
        sys.exit( 1 )

# DEFAULT PARAMETERS
DATASET = 'Dataset/test'
PREDICTION_LB = 'result/mobilenet_400_test.txt'
ODIR = 'test_mobilenet_400'

# LOAD PARAMETER VALUES
i = 1
while i < len( sys.argv ) :
    cmd = str( sys.argv[ i ] )
    i = i + 1
    if cmd == '--dataset' :
        DATASET = str( sys.argv[ i ] )
    elif cmd == '--predict' :
        PREDICTION_LB = str( sys.argv[ i ] )
    elif cmd == '--odir' :
        ODIR = str( sys.argv[ i ] )
    i = i + 1

print('Dataset           : ', DATASET)
print('Prediction labels : ', PREDICTION_LB)
print('Output directory  : ', ODIR)


def loadNames( path ) :
    cmd = path + "data.txt"
    f = open( cmd, 'r' )
    names = []
    ref_lb = []
    for line in f :
        row = line.split()
        names.append( path + str( row[ 0 ] ) )
        ref_lb.append( int( row[ 1 ] ) )
    f.close()
    return names, ref_lb

def loadLabels( file ) :
    f = open( file, 'r' )
    lb = []
    for line in f :
        row = line.split()
        lb.append( int( row[ 0 ] ) )
    f.close()
    return lb

names, ref_lb = loadNames( DATASET + "/" )
lb = loadLabels( PREDICTION_LB )
os.system( 'mkdir ' + ODIR  )
for i in range( len( names ) ) :
    img = cv2.imread( names[ i ] )
    if lb[ i ] == 1 :
        img = cv2.rectangle(img, (0,0) , (230, 80), (0, 0, 255) , -1)
        img = cv2.putText( img, 'Obstacle', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA )
    else :
        img = cv2.rectangle(img, (0,0) , (350, 80), (50, 205, 50), -1)
        img = cv2.putText( img, 'Non-obstacle', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_AA )
    obs = 'obs' if ref_lb[ i ] else 'non-obs'
    new_name = names[ i ].replace(DATASET+"/","")
    new_name = new_name.replace('non_obs/','')
    new_name = new_name.replace('obs/','')
    new_name = new_name.replace('test/','')
    if ref_lb[ i ] == lb[ i ] :
        cmd = ODIR  + '/ok_ref_' + obs + '_' + new_name
    else :
        cmd = ODIR + '/bad_ref_' + obs + '_' +  new_name
    print( cmd )
    cv2.imwrite( cmd, img )
