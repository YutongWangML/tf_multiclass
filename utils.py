from keras.utils import to_categorical
import keras.backend as K


def to_t_categorical(y_raw,num_classes=None, dtype='float32'):
    """
    convert raw labels 
    e.g., [0,1,2,...]
    to trimmed categorical label encodings
    y_raw here refers to the labels that takes values in {0, 1,..., n_classes}
    """
    return to_categorical(y_raw, num_classes, dtype)[:,:-1]


def from_t_categorical(y):
    """ convert trimmed categorical label encodings to raw labels
    """
    
    n_classes_m1 = y.shape[1] # number of classes minus 1
    
    A = K.cast(K.all(y==0,axis=1), K.floatx())
    return (n_classes_m1)*A + K.cast(K.argmax(y,axis=1), K.floatx())*(1.-A)




def predict_classes_from_r_margin(y):
    """ predict classes from relative margin
    """
    
    n_classes_m1 = y.shape[1] # number of classes minus 1
    
    A = K.cast(K.all(y>=0,axis=1),dtype = K.floatx())
    return  A*n_classes_m1 + (1.-A)*K.cast(K.argmin(y,axis=1),K.floatx())
