import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class ARKDRegressor:
    """Nonnegative mixture of global, local and residual PSD kernels.

    The local kernel uses fixed input-space gates; the residual kernel uses
    a centered nonlinear feature map. Both remain PSD out of sample.
    """
    def __init__(self, alpha=0.1, gamma_global=0.1, gamma_local=1.0,
                 gamma_residual=0.5, n_regions=4, random_state=0):
        self.alpha = alpha
        self.gamma_global = gamma_global
        self.gamma_local = gamma_local
        self.gamma_residual = gamma_residual
        self.n_regions = n_regions
        self.random_state = random_state

    def _gates(self, X):
        D = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        Z = -self.gamma_local * D
        Z -= Z.max(axis=1, keepdims=True)
        W = np.exp(Z)
        return W / W.sum(axis=1, keepdims=True)

    def _components(self, X, Y):
        global_k = rbf_kernel(X, Y, gamma=self.gamma_global)
        gates_x, gates_y = self._gates(X), self._gates(Y)
        local_k = rbf_kernel(X, Y, gamma=self.gamma_local) * (gates_x @ gates_y.T)
        # Center a nonlinear RBF feature map relative to the training anchors.
        # Feature inner products are PSD, including for held-out points.
        fx = rbf_kernel(X, self.anchors_, gamma=self.gamma_residual) - self.anchor_mean_
        fy = rbf_kernel(Y, self.anchors_, gamma=self.gamma_residual) - self.anchor_mean_
        residual_k = (fx @ fy.T) / len(self.anchors_)
        return np.stack([global_k, local_k, residual_k], axis=0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.scaler_ = StandardScaler().fit(X)
        self.X_ = self.scaler_.transform(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=min(self.n_regions, len(X)), replace=False)
        self.centers_ = self.X_[idx]
        self.anchors_ = self.X_.copy()
        self.anchor_mean_ = rbf_kernel(self.anchors_, self.anchors_,
                                      gamma=self.gamma_residual).mean(axis=0)
        components = self._components(self.X_, self.X_)
        # Select weights by out-of-fold prediction error; folds use the same
        # unsupervised feature map, fitted only from X (not from y).
        folds = list(KFold(n_splits=min(5, len(X)), shuffle=True,
                           random_state=self.random_state).split(self.X_))

        def loss(weights):
            K = np.tensordot(weights, components, axes=(0, 0))
            errors = []
            for train, val in folds:
                model = KernelRidge(alpha=self.alpha, kernel='precomputed')
                model.fit(K[np.ix_(train, train)], y[train])
                pred = model.predict(K[np.ix_(val, train)])
                errors.extend((pred - y[val]) ** 2)
            return float(np.mean(errors))

        result = minimize(loss, np.ones(3) / 3, method='SLSQP',
                          bounds=[(0, 1)] * 3,
                          constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                          options={'maxiter': 100, 'ftol': 1e-8})
        self.weights_ = result.x if result.success else np.ones(3) / 3
        self.weights_ = np.maximum(self.weights_, 0)
        self.weights_ /= self.weights_.sum()
        K = np.tensordot(self.weights_, components, axes=(0, 0))
        self.model_ = KernelRidge(alpha=self.alpha, kernel='precomputed').fit(K, y)
        return self

    def predict(self, X):
        X = self.scaler_.transform(np.asarray(X, dtype=float))
        components = self._components(X, self.X_)
        K = np.tensordot(self.weights_, components, axes=(0, 0))
        return self.model_.predict(K)


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, (160, 2))
    y = np.sin(2 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 160)
    train, test = np.arange(120), np.arange(120, 160)
    model = ARKDRegressor().fit(X[train], y[train])
    prediction = model.predict(X[test])
    print('Weights [global, local, residual]:', np.round(model.weights_, 3))
    print('Test RMSE:', round(np.sqrt(mean_squared_error(y[test], prediction)), 4))