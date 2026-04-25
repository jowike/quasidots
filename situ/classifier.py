import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.validation import check_random_state

from .kernels import transform


class SITUClassifier:
    def __init__(self, num_kernels=1000, operation="AT", random_state=None):
        self.num_kernels = num_kernels
        self.operation = operation
        self.random_state = random_state

    def _op_to_int(self):
        mapping = {
            "AT": 0,
            "ES": 1,
            "SM": 2,
            "SR": 3,
            "LE": 4,
            "LA": 5
        }
        return mapping[self.operation]

    def _generate_kernels(self, n_timepoints, n_channels):
        rng = np.random.RandomState(self.random_state)

        lengths = rng.choice([7, 9, 11], self.num_kernels).astype(np.int32)

        num_channel_indices = np.zeros(self.num_kernels, dtype=np.int32)

        for i in range(self.num_kernels):
            limit = min(n_channels, lengths[i])
            num_channel_indices[i] = int(
                2 ** rng.uniform(0, np.log2(limit + 1))
            )

        channel_indices = rng.randint(
            0, n_channels, size=num_channel_indices.sum()
        ).astype(np.int32)

        weights = rng.normal(
            0, 1,
            int(np.dot(lengths, num_channel_indices))
        ).astype(np.float32)

        biases = rng.uniform(-1, 1, self.num_kernels).astype(np.float32)

        dilations = np.zeros(self.num_kernels, dtype=np.int32)
        paddings = np.zeros(self.num_kernels, dtype=np.int32)

        for i in range(self.num_kernels):
            dilation = 2 ** rng.uniform(
                0, np.log2((n_timepoints - 1) / (lengths[i] - 1))
            )
            dilation = int(dilation)

            dilations[i] = dilation
            paddings[i] = (
                ((lengths[i] - 1) * dilation) // 2
                if rng.randint(2) else 0
            )

        return (
            weights,
            lengths,
            biases,
            dilations,
            paddings,
            num_channel_indices,
            channel_indices,
        )

    def fit(self, X, y):
        X = np.array(X)

        n_instances, n_channels, n_timepoints = X.shape

        self.kernels = self._generate_kernels(n_timepoints, n_channels)

        op = self._op_to_int()

        X_transformed = transform(X, *self.kernels, op)

        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(X_transformed, y)

        return self

    def predict(self, X):
        X = np.array(X)
        op = self._op_to_int()

        X_transformed = transform(X, *self.kernels, op)

        return self.classifier.predict(X_transformed)