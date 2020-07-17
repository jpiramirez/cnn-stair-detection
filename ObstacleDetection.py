import cv2
import os

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

model = 1000
model_name = 'fusion'
mode = 'train' # test, validation, train

folder = 'Dataset/' + mode + '/'
names, ref_lb = loadNames( folder )
lb = loadLabels( 'models/mobilenet_'+ model_name + '_' + str( model ) + '_' + mode + '.txt' )
path = mode + '_' + model_name + '_' + str( model ) + '/'   # Create output directory
cmd = 'mkdir ' + path
os.system( cmd )

for i in range( len( names ) ) :
    img = cv2.imread( names[ i ] )
    if lb[ i ] == 1 :
        img = cv2.rectangle(img, (5,10) , (150, 50), (0, 0, 255) , -1) # start_point (top left), end_point (bottom right), color (BGR), thickness( fill -1)
        img = cv2.putText( img, 'Obstacle', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA )
    else :
        img = cv2.rectangle(img, (5,10) , (230, 50), (0, 255, 0), -1)
        img = cv2.putText( img, 'Non-obstacle', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA )
    obs = 'obs' if ref_lb[ i ] else 'non-obs'
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

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
