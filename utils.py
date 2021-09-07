from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf

def add_constant_column(x,val):
    return tf.concat([x,tf.constant(val,shape=(x.shape[0],1))],axis=-1)
#     return K.concatenate([x,K.constant(val,shape=(x.shape[0],1))],axis=-1)



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
    
#     n_classes_m1 = y.shape[1] # number of classes minus 1
    
#     A = K.cast(K.all(y==0,axis=1), K.floatx())
#     return (n_classes_m1)*A + K.cast(K.argmax(y,axis=1), K.floatx())*(1.-A)

    return K.cast_to_floatx(tf.argmax(add_constant_column(y, 0.5),axis=1))

#     return K.cast_to_floatx(K.argmax(add_constant_column2(y, 0.5),-1))




def predict_classes_from_r_margin(z):
    """ predict classes from relative margin
    """
    return K.cast_to_floatx(tf.argmin(add_constant_column(z,0.0),axis=1))
    
#     n_classes_m1 = z.shape[1] # number of classes minus 1
    
#     A = K.cast(K.all(z>=0,axis=1),dtype = K.floatx())
#     return  A*n_classes_m1 + (1.-A)*K.cast(K.argmin(z,axis=1),K.floatx())
