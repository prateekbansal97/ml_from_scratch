import numpy as np

def ReLu(X):
    return np.maximum(0, X)

def softmax(X, clip=True):
    maxa = np.max(X, axis=1, keepdims=True)
    X = X - maxa
    probs = np.exp(X)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    if clip:
        probs = np.clip(probs, 1e-10, 1.0)
    return probs
