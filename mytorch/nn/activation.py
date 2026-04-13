import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        dim = self.dim if self.dim >= 0 else self.dim + Z.ndim

        Z_moved = np.moveaxis(Z, dim, -1)
        orig_shape = Z_moved.shape
        Z_2d = Z_moved.reshape(-1, orig_shape[-1])

        # Numerically stable softmax
        Z_shifted = Z_2d - np.max(Z_2d, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        A_2d = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        A_moved = A_2d.reshape(orig_shape)
        self.A = np.moveaxis(A_moved, -1, dim)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        dim = self.dim if self.dim >= 0 else self.dim + dLdA.ndim

        A_moved = np.moveaxis(self.A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)

        orig_shape = A_moved.shape
        A_2d = A_moved.reshape(-1, orig_shape[-1])
        dLdA_2d = dLdA_moved.reshape(-1, orig_shape[-1])

        # Vectorized Jacobian-vector product for softmax
        dot = np.sum(dLdA_2d * A_2d, axis=1, keepdims=True)
        dLdZ_2d = A_2d * (dLdA_2d - dot)

        dLdZ_moved = dLdZ_2d.reshape(orig_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        return dLdZ
 

    