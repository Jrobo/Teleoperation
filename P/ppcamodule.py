import numpy as np
from numpy.linalg import svd

class PPCAModule:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None
        self.mu = None
        self.sigma = None

    def fit(self, X):
        """
        Fit PPCA model to the observed joint velocities.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Observed joint velocities.
        """
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu

        _, S, Vt = svd(X_centered.T @ X_centered / X.shape[0])

        self.W = Vt[:self.n_components, :].T
        self.sigma = np.sqrt(np.maximum(0, np.sum(S[self.n_components:]) / (X.shape[0] - self.n_components)))

    def transform(self, A):
        """
        Transform joystick actions to joint velocities using PPCA.

        Parameters:
        - A: numpy array, shape (n_samples, n_actions)
            Joystick actions.

        Returns:
        - Y: numpy array, shape (n_samples, n_features)
            Predicted joint velocities.
        """
        if A.shape[1] != self.W.shape[1]:
            raise ValueError("The number of features in joystick actions must match the trained PPCA model.")

        if A.shape[0] == 0:
            raise ValueError("No joystick actions provided.")

        Y = A @ self.W

        if self.sigma > 0:
            Y += np.random.normal(0, self.sigma, Y.shape)  # Add Gaussian noise

        return Y + self.mu
