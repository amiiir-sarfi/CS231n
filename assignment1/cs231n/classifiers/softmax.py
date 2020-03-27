from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = X@W #N, C
    regularization = np.sum(W**2)
    for i in range(X.shape[0]):
      exps = np.exp(y_pred[i] - np.max(y_pred[i]))
      exps /= np.sum(exps)
      p = exps[y[i]]
      loss += -np.log(p)
      for j in range(W.shape[1]):
        dW[:,j] += X[i] * exps[j] #N x D        DxC
      dW[:,y[i]] -= X[i]

    loss /= X.shape[0]
    dW /= X.shape[0]
    
    loss += reg*regularization
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = X@W #
    exps = np.exp(y_pred-np.max(y_pred,1)[:,np.newaxis])
    exps /= np.sum(exps,-1)[:,np.newaxis]
    p = exps[range(exps.shape[0]),y]
    loss = np.sum(-np.log(p))
    loss /= X.shape[0]
    dW = X.T@exps
    dW /= X.shape[0]
    regularization = np.sum(W**2)
    dW += reg * 2 * W 
    loss += reg*regularization

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
