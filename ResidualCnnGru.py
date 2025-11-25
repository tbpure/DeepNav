import tensorflow as tf

def build_cnn_residual_gru_sequential(input_shape=(None, 10), output_dim=6):
    inputs = tf.keras.Input(shape=input_shape)

    # CNN + 残差
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    shortcut = tf.keras.layers.Conv1D(64, 1, padding='same')(inputs)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    # GRU部分
    x = tf.keras.layers.GRU(20, return_sequences=True)(x)
    x = tf.keras.layers.GRU(200, return_sequences=True)(x)
    x = tf.keras.layers.GRU(200, return_sequences=True)(x)
    x = tf.keras.layers.GRU(20, return_sequences=False)(x)
    outputs = tf.keras.layers.Dense(output_dim)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ResCNN_GRU')
    return model