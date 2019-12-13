from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Add
from keras.layers.merge import add
from keras.activations import relu, softmax
from keras import regularizers
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model

def get_actor_model():
    model = Sequential()
    states = Input(shape = (20, ))
    x = Dense(400)(states)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Dense(1, activation = 'tanh')(x)

    model = Model(inputs = states, outputs = x)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_critic_model():
    states = Input(shape = (20, ))
    actions = Input(shape = (1, ))

    x = Dense(400, activation = 'relu')(states)
    x = Add()([x, actions])
    x = Dense(300, activation = 'relu', input_shape = x.shape)(x)
    x = Dense(1, activation = 'linear')(x)

    model = Model(inputs = [states, actions], outputs = x)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


actor_model = get_actor_model()
critic_model = get_critic_model()

plot_model(actor_model, to_file='actor_model.png', show_shapes=True, show_layer_names=True)
plot_model(critic_model, to_file='critic_model.png', show_shapes=True, show_layer_names=True)