import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_mobilenet_model( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape,include_top = False,weights = 'imagenet')
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense( 1, activation = 'sigmoid' )
    model = keras.Sequential([ base_model, global_average_layer,prediction_layer])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),                     # 500-599 (1e-9), 600-699 (1e-10)
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def make_mobilenet_fusion( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape, include_top = False,weights = 'imagenet')
    #print("Number of layers in the base model: ", len(base_model.layers))
    #base_model.trainable = True
    #fine_tune_at = 80  # Fine-tune from this layer onwards
    #for layer in base_model.layers[:fine_tune_at]: # Freeze all the layers before the `fine_tune_at` layer
    #    layer.trainable =  False
    img_input = keras.Input(shape= myshape, name="img_input")
    x = base_model( img_input )
    x = layers.GlobalAveragePooling2D()( x )
    imu_input = keras.Input( shape = (6), name="imu_input")
    x = tf.concat([x, imu_input], 1)
    prediction = layers.Dense(1, activation='sigmoid', name='prediction')( x )
    model = keras.Model(
        inputs=[img_input, imu_input],
        outputs=[ prediction ],
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4),                        # 0-99 (1e-4), 100-199 (1e-5), 200-299 (1e-6), 300-399 (1e-7), so on
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
