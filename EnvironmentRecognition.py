import cv2
import os

def loadNames( path ) :
    cmd = path + "data_3class.txt"
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

model = 500
model_name = 'mobilenet_3class' # fusion, mobilenet
mode = 'test' # test, validation, train

folder = 'Dataset/' + mode + '/'
names, ref_lb = loadNames( folder )

cmd = './../backup_models/08.21.2020/' + model_name + '_';
cmd = cmd + str( model ) + '_' + mode + '.txt'
lb = loadLabels( cmd )
path = mode + '_' + model_name + '_' + str( model ) + '/'   # Create output directory

cmd = 'mkdir ' + path
os.system( cmd )

for i in range( len( names ) ) :
    img = cv2.imread( names[ i ] )
    if lb[ i ] == 1 :
        img = cv2.rectangle(img, (0,0) , (340, 80), (0, 0, 255) , -1) # start_point (top left), end_point (bottom right), color (BGR), thickness( fill -1)
        img = cv2.putText( img, 'Stair ascent', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA )
    elif lb[ i ] == 2 :
        img = cv2.rectangle(img, (0,0) , (340, 80), (0, 128, 255) , -1)
        img = cv2.putText( img, 'Stair descent', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA )
    else :
        img = cv2.rectangle(img, (0,0) , (340, 80), (50, 205, 50), -1)
        img = cv2.putText( img, 'level-ground', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_AA )


    if ref_lb[ i ] == 1 :
        obs = 'ascent'
    elif ref_lb[ i ] == 2 :
        obs = 'descent'
    else :
        obs = 'ground'

    new_name = names[ i ].replace(folder,"")
    new_name = new_name.replace('non_obs/','')
    new_name = new_name.replace('obs/','')
    new_name = new_name.replace('test/','')

    if ref_lb[ i ] == lb[ i ] :
        cmd = path + 'ok_ref_' + obs + '_' + new_name
    else :
        cmd = path + 'bad_ref_' + obs + '_' +  new_name
    print( cmd )

    cv2.imwrite( cmd, img )
