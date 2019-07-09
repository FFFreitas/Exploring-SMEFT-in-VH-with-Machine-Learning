from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


def get_data_ZH_0L_cHW_0d01():
    with np.load('../charanjit_data/0L/bsmlike_processed/X_train_0L_cHW_0d01.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/0L/bsmlike_processed/X_test_0L_cHW_0d01.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
        
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_0L_cHW_0d03():
    with np.load('../charanjit_data/0L/bsmlike_processed/X_train_0L_cHW_0d03.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/0L/bsmlike_processed/X_test_0L_cHW_0d03.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
        
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_0L_cHW_0d1():
    with np.load('../charanjit_data/0L/bsmlike_processed/X_train_0L_cHW_0d1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/0L/bsmlike_processed/X_test_0L_cHW_0d1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_0L_cHW_1():
    with np.load('../charanjit_data/0L/bsmlike_processed/X_train_0L_cHW_1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/0L/bsmlike_processed/X_test_0L_cHW_1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
        
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)


def get_data_WH_1L_cHW_0d01():
    with np.load('../charanjit_data/1L/bsmlike_processed/X_train_1L_cHW_0d01.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/1L/bsmlike_processed/X_test_1L_cHW_0d01.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_WH_1L_cHW_0d03():
    with np.load('../charanjit_data/1L/bsmlike_processed/X_train_1L_cHW_0d03.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/1L/bsmlike_processed/X_test_1L_cHW_0d03.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_WH_1L_cHW_0d1():
    with np.load('../charanjit_data/1L/bsmlike_processed/X_train_1L_cHW_0d1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/1L/bsmlike_processed/X_test_1L_cHW_0d1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_WH_1L_cHW_1():
    with np.load('../charanjit_data/1L/bsmlike_processed/X_train_1L_cHW_1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/1L/bsmlike_processed/X_test_1L_cHW_1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_2L_cHW_0d01():
    with np.load('../charanjit_data/2L/bsmlike_processed/X_train_2L_cHW_0d01.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/2L/bsmlike_processed/X_test_2L_cHW_0d01.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    

    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_2L_cHW_0d03():
    with np.load('../charanjit_data/2L/bsmlike_processed/X_train_2L_cHW_0d03.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/2L/bsmlike_processed/X_test_2L_cHW_0d03.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_ZH_2L_cHW_0d1():
    with np.load('../charanjit_data/2L/bsmlike_processed/X_train_2L_cHW_0d1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/2L/bsmlike_processed/X_test_2L_cHW_0d1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    

    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)


def get_data_ZH_2L_cHW_1():
    with np.load('../charanjit_data/2L/bsmlike_processed/X_train_2L_cHW_1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/2L/bsmlike_processed/X_test_2L_cHW_1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    

    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_all_cHW_0d01():
    with np.load('../charanjit_data/3_channels/bsmlike/X_train_all_cHW_0d01.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/3_channels/bsmlike/X_test_all_cHW_0d01.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_all_cHW_0d03():
    with np.load('../charanjit_data/3_channels/bsmlike/X_train_all_cHW_0d03.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/3_channels/bsmlike/X_test_all_cHW_0d03.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
 
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_all_cHW_0d1():
    with np.load('../charanjit_data/3_channels/bsmlike/X_train_all_cHW_0d1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/3_channels/bsmlike/X_test_all_cHW_0d1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    
    #y_train = to_categorical(y_train, nb_classes)
    #y_test = to_categorical(y_test, nb_classes)
    
    
    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)

def get_data_all_cHW_1():
    with np.load('../charanjit_data/3_channels/bsmlike/X_train_all_cHW_1.npz') as f:
        X_train, y_train = f['x'], f['y']    

    with np.load('../charanjit_data/3_channels/bsmlike/X_test_all_cHW_1.npz') as f:
        X_test, y_test = f['x'], f['y']    
        
    nb_classes = 2
    batch_size = 256
    input_shape = (X_train.shape[1],)
    

    return (nb_classes, batch_size, input_shape, X_train, X_test, y_train, y_test)


def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.
    Args:
        network (dict): the parameters of the network
    Returns:
        a compiled network.
    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    if dataset == 'ZH_0L_cHW_0.01':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_0L_cHW_0d01()
######################################################################    
    elif dataset == 'ZH_0L_cHW_0.03':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_0L_cHW_0d03()
######################################################################
    elif dataset == 'ZH_0L_cHW_0.1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_0L_cHW_0d1()
######################################################################
    elif dataset == 'ZH_0L_cHW_1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_0L_cHW_1()
######################################################################    
    elif dataset == 'WH_1L_cHW_0.01':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_WH_1L_cHW_0d01()
######################################################################
    elif dataset == 'WH_1L_cHW_0.03':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_WH_1L_cHW_0d03()
######################################################################
    elif dataset == 'WH_1L_cHW_0.1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_WH_1L_cHW_0d1()
######################################################################
    elif dataset == 'WH_1L_cHW_1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_WH_1L_cHW_1()
######################################################################        
    elif dataset == 'ZH_2L_cHW_0.01':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_2L_cHW_0d01()
######################################################################
    elif dataset == 'ZH_2L_cHW_0.03':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_2L_cHW_0d03()
######################################################################
    elif dataset == 'ZH_2L_cHW_0.1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_2L_cHW_0d1()
######################################################################
    elif dataset == 'ZH_2L_cHW_1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_ZH_2L_cHW_1()
######################################################################
    elif dataset == 'all_cHW_0.01':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_all_cHW_0d01()
######################################################################
    elif dataset == 'all_cHW_0.03':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_all_cHW_0d03()
######################################################################
    elif dataset == 'all_cHW_0.1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_all_cHW_0d1()
######################################################################
    elif dataset == 'all_cHW_1':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_data_all_cHW_1()
######################################################################        
    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1] # 1 is accuracy. 0 is loss.
