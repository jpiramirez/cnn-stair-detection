def loadNames( file ) :
    f = open( file, 'r' )
    for line in f :
        row = line.split()
        for i in range( len(row ) ) :
            if row[ i ] ==  'Epoch' :
                epoch = row[ i + 1 ].replace( '/1000', '' )
                print( int( epoch), end= ' ' )
            elif row[ i ] ==  'accuracy:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] == 'val_accuracy:' :
                print( float(row[ i + 1 ] ) )
    f.close()
    return 0

a = loadNames( 'fusion_train_evo_raw.txt' )
