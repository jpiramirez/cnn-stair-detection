
def loadNames( file ) :
    f = open( file, 'r' )
    for line in f :
        row = line.split()
        if( len(row) != 2 and len(row) != 17 ) :
            continue;
        for i in range( len(row ) ) :
            if row[ i ] ==  'Epoch' :
                epoch = row[ i + 1 ].replace( '/500', '' )
                print( int( epoch ), end= ' ' )
            elif row[ i ] == 'loss:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] == 'val_loss:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] ==  'accuracy:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] == 'val_accuracy:' :
                print( float(row[ i + 1 ] ) )
    f.close()
    return 0

a = loadNames( 'fusion_evolution_raw.txt' )
