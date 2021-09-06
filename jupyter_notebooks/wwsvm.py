from scipy.sparse import coo_matrix, eye
import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import linear_kernel


def get_involution_code(n_classes):
    # involution code for ncm1-ary multiclass classification 
    
    # number of classes minus 1
    ncm1 = n_classes-1
    
    # the Involution code
    ic = []
    for j in range(ncm1):
        row_indices = [i for i in range(ncm1) if i != j] + [i for i in range(ncm1)]
        col_indices = [i for i in range(ncm1) if i != j] + [j for i in range(ncm1)]
        values = [1.0 for i in range(ncm1) if i != j] + [-1.0 for i in range(ncm1)]
        rhoj = coo_matrix((values,(row_indices, col_indices))).todense()
        ic.append(rhoj)
    ic.append(eye(ncm1).todense())

    # The all-ones plus identity matrix
    Theta = np.ones((ncm1,ncm1)) + np.eye(ncm1)


    # The Pi matrix
    Pi = np.concatenate((np.ones((ncm1,1)).T, -np.eye(ncm1))).T
    
    return ic, Theta, Pi


class WWSVM:
    def __init__(self, C=1):
        self.C = C
        self.solvers = {'cvxpy': self._fit_cvxpy}
        
    def get_multiclass_kernel(self,K, y, n_classes):
        # K = (ordinary) kernel, i.e., K[i][j] = kernel_function(x[i,:], x[j,:])
        #     K.shape = (n_samples, n_samples)
        # y = labels
        #     y.shape = (n_samples,)
        # n_classes = number of classes
        #     Usually, n_classes = len(np.unique(y))
        #     However, we only require that max(y) < n_classes
        n_samples = len(y)
        return np.concatenate( [np.concatenate([ K[i,j] * self.ic[y[i]] @ self.Theta @ self.ic[y[j]].T 
                                                for i in range(n_samples)]) 
                                for j in range(n_samples)],
                              axis=1)
    
    def fit(self, X, y, method = 'cvxpy'):
        self.n_classes = len(np.unique(y))
        self.ic, self.Theta, self.Pi = get_involution_code(self.n_classes)
        self.X_train = X
        self.y_train = y
        
        # self.beta is computed here:
        self.solvers[method](X, y)
        
        self.B = np.concatenate([ self.beta[i,:] @ self.ic[y[i]] @ self.Pi for i in range(self.beta.shape[0]) ])
        return None
        
        

        
    def predict(self,X_test):
        
        return np.asarray(np.argmax(linear_kernel(X_test, self.X_train) @ self.B,
                                    axis=1)).flatten()

        
    def _fit_cvxpy(self, X, y):
        # solve the dual formulation of WW-SVM 
        # via convex quadratic program solver cvxpy
        K = linear_kernel(X)

        Q = self.get_multiclass_kernel(K, y, self.n_classes)
        q = np.ones(((self.n_classes-1)*len(y),))
        
        x = cp.Variable(len(q))
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, Q) - q.T @ x),
                         [ x <= self.C,
                           x >= 0] )
        prob.solve()
        
        self.dual_objective_exact = -prob.value
        self.beta = np.reshape(x.value,(X.shape[0],self.n_classes-1))
        return None