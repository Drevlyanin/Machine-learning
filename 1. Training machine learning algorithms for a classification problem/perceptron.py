import numpy as np

class Perceptron(object):
    """perceptron-based classifier.
    
    Options
    --------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes through the training data set.
        
    Attributes
    --------
    w_ : dimensional array
        Weighting coefficients after adjustment.
    errors_ : list
        Number of misclassification cases in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """ Fit the model to the training data.
        
        Options
        --------
        X: {array-like}, shape = [n_samples, n_features]
            training vectors, where
            n_samples - number of samples and
            n_features - number of features.
        y: {array-like}, shape [n_samples]
            Target values.
            
        Returns
        --------
        self: object
        """
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """ Calculate input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        """ Return class label after single hop """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
