import abc
from typing import List, Optional

import numpy as np


def chunks(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, :]


class Loss(object):
    @abc.abstractmethod
    def __call__(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gradient(self, true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def __call__(self, true, predicted):
        return np.sqrt(((true - predicted) ** 2).sum())

    def gradient(self, true, predicted):
        return 2 * (predicted - true)


class Layer(object):
    def __init__(self):
        self._last_input: Optional[np.ndarray] = None

    @abc.abstractmethod
    def forward(self, _input: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, gradient: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass


class NeuralNetwork(object):
    def __init__(self):
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, _input: np.ndarray) -> np.ndarray:
        last_layer_output = _input
        for layer in self.layers:
            last_layer_output = layer.forward(last_layer_output)

        return last_layer_output

    def backward(self, gradient: np.ndarray, batch_size: int, learning_rate: float):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, batch_size=batch_size, learning_rate=learning_rate)

    # noinspection PyPep8Naming
    def train(self, X: np.ndarray, Y: np.ndarray, loss: Loss, epochs: int = 5, batch_size: int = 1,
              learning_rate: float = 0.001) -> List[float]:
        loss_history: List[float] = []

        for epoch in range(epochs):
            for X_batch, y_batch in zip(chunks(X, batch_size), chunks(Y, batch_size)):
                assert X_batch.shape == (batch_size, X.shape[1])
                assert y_batch.shape == (batch_size, Y.shape[1])

                y_prediction = self.forward(X_batch)

                assert y_prediction.shape == (batch_size, Y.shape[1])

                loss_value = loss(y_batch, y_prediction)
                loss_gradient_value = loss.gradient(y_batch, y_prediction)

                self.backward(loss_gradient_value, batch_size, learning_rate)

                loss_history.append(float(loss_value))

        return loss_history


class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim + 1  # bias
        self.output_dim = output_dim

        # Xavier initialization
        self.weights: np.ndarray = (np.random.randn(self.input_dim, self.output_dim) - 0.5) * np.sqrt(
            1 / self.input_dim)

    def forward(self, _input: np.ndarray) -> np.ndarray:
        # Extending input with bias
        last_input = np.ones((_input.shape[0], _input.shape[1] + 1))
        last_input[:, :-1] = _input
        self._last_input = last_input

        return last_input @ self.weights

    def backward(self, gradient: np.ndarray, *args, **kwargs) -> np.ndarray:
        batch_size = kwargs.get('batch_size')
        learning_rate = kwargs.get('learning_rate')

        assert self._last_input.shape == (batch_size, self.input_dim)
        assert gradient.shape == (batch_size, self.output_dim), gradient.shape

        weights_update = np.matmul(self._last_input[:, :, np.newaxis], gradient[:, np.newaxis, :]).transpose((1, 2, 0))
        assert weights_update.shape == (self.input_dim, self.output_dim, batch_size)

        weights_update = weights_update.mean(axis=2)
        assert weights_update.shape == (self.input_dim, self.output_dim)

        output_gradient = (gradient @ self.weights.T)[:, :-1]
        assert output_gradient.shape == (batch_size, self.input_dim - 1)

        # Updating weights
        self.weights -= learning_rate * weights_update

        return output_gradient


class Sigmoid(Layer):
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.exp(x) / (1 + np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def forward(self, _input: np.ndarray) -> np.ndarray:
        self._last_input = _input

        return self._sigmoid(self._last_input)

    def backward(self, gradient: np.ndarray, *args, **kwargs) -> np.ndarray:
        return gradient * self.derivative(self._last_input)


class ReLU(Layer):
    def forward(self, _input: np.ndarray) -> np.ndarray:
        self._last_input = _input.copy()

        return _input * (_input > 0)

    def backward(self, gradient: np.ndarray, *args, **kwargs) -> np.ndarray:
        return gradient * (self._last_input > 0)
