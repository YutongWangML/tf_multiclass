from tensorflow.python.keras.utils import losses_utils
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
import tensorflow as tf

def get_relative_margins(y_true, y_pred):
    # Output: the matrix 
    #     z_pred = A - B - C 
    # such that
    #     1. z_pred.shape == y_pred.shape
    #     2. z_pred[i,:] == y_pred[i,:] * rho[y_true[i]].T
    #         where rho[j] is the j-th involution code
    # Inputs: 
    #     y_true = to_category(y_true_raw)[:,:-1]
    #
    # y_true_raw here refers to the labels that takes values in {0, 1,..., n_classes}
    # the raw labels are then transformed to one-hot representation and the last entry is dropped
    # 
    #     y_pred = real-valued output of a model whose final layer has n_classes units
    A = y_pred
    C = y_pred*y_true
    B = K.expand_dims(K.sum(C,axis=1),axis=1)
    return A - B - C


class PERMLoss(Loss):
    def __init__(self,fn,reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.fn = fn

    def call(self, y_true, y_pred):
        """
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        z = get_relative_margins(y_true, y_pred)
        return self.fn(z)



def cs_hinge(z):
    # Crammer-Singer hinge function
    return K.max(K.maximum(1-z,0.),axis=1)


class CSHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='cs_hinge'):
        super(CSHinge, self).__init__(cs_hinge, name=name, reduction=reduction)



def ww_hinge(z):
    # Weston-Watkins hinge function
    return K.sum(K.maximum(1-z,0.),axis=1)


class WWHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='ww_hinge'):
        super(WWHinge, self).__init__(ww_hinge, name=name, reduction=reduction)


        
def dkr_hinge(z):    
    # Duchi-Khosravi-Ruan hinge function
    z0 = tf.pad( z, [[0,0],[0,1]] ) # add a column of zeros to z
    z0sorted = -tf.nn.top_k(-z0,k = z0.shape[1], sorted=True).values
    z0sorted_csummed = K.cumsum(z0sorted,axis=1)+1
    
    ell = K.arange(1,z0.shape[1]+1)
    ell_inv = K.cast_to_floatx(K.reshape(1/ell,[1,z0.shape[1]]))
    
    z0sorted_csummed_scaled = z0sorted_csummed * ell_inv

    return 1-K.min(z0sorted_csummed_scaled,axis=1)


class DKRHinge(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='dkr_hinge'):
        super(DKRHinge, self).__init__(dkr_hinge, name=name, reduction=reduction)

        

def cross_entropy(z):
    # Cross Entropy implemented in the PERM loss framework
    return K.log(1+K.sum(K.exp(-z),axis=1))


class CrossEntropy(PERMLoss):
    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='crossentropy'):
        super(CrossEntropy, self).__init__(cross_entropy, name=name, reduction=reduction)