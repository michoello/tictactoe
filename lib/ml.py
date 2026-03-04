from __future__ import annotations
import math
import json
import random
from typing import Any, Optional, Union


# Returns matrix filled with uniform distribution within [-k;k]
def random_matrix(m: int, n: int, k: float = 1.0) -> list[list[float]]:
    return [[ (random.random() * 2 - 1) * k for _ in range(n)] for _ in range(m)]


# Sigmoid activation and its derivative
def sigmoid(x: float) -> float:
    # Clipping to prevent overflow when batching gradients
    # x = max(min(x, 20), -20)
    # print("AAAA", x)
    return 1 / (1 + math.exp(-x))


def vectorized(f: Any, x: list[list[float]]) -> list[list[float]]:
    return [[f(v) for v in row] for row in x]


def sigmoid_derivative(x: list[list[float]]) -> list[list[float]]:
    return [[v * (1 - v) for v in row] for row in vectorized(sigmoid, x)]


# Transpose
def transpose(A: list[list[float]]) -> Any:
    return list(zip(*A))


# Matrix multiplication
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(a * b for a, b in zip(row, col)) for col in transpose(B)] for row in A]


# Element-wise addition
def add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


# Element-wise subtraction
def subtract(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


# Element-wise multiplication
def elementwise_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def mul(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    return [[element * scalar for element in row] for row in matrix]


def _dims(v: Any) -> list[int]:
    if not isinstance(v, list):
        return []
    dimension = [len(v)]
    if len(v) > 0 and isinstance(v[0], list):
        dimension.extend(_dims(v[0]))
    return dimension


class BB:
    args: list[Any]
    value: Optional[list[list[float]]]
    dvalue: Optional[list[list[float]]]
    def __init__(self, *args: Any) -> None:
        self.args = list(args)
        self.value = None
        self.dvalue = None

    def dims(self) -> list[int]:
        if self.value is None:
            self.val()
        return _dims(self.value)

    def val(self) -> Any:
        self.value = self.args[0]
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        # why does this happen even without batching gradients?
        # if self.dvalue is None:
        self.dvalue = dvalue
        # else:
        #   self.dvalue = add(self.dvalue, dvalue)

    def dval(self) -> Any:
        return self.dvalue

    def arg(self, i: int) -> Any:
        return self.args[i]

    # Only for inputs and weights. TODO: better API
    def set(self, newarg: Any) -> None:
        self.args[0] = newarg

    def appl(self, learning_rate: float) -> None:
        assert self.value is not None
        assert self.dvalue is not None
        self.value = subtract(self.value, mul(self.dvalue, learning_rate))
        self.dvalue = None
        self.args[0] = self.value

    def __matmul__(self, other: Any) -> BBMatmul:
        return BBMatmul(self, other)

    def __add__(self, other: Any) -> BBSum:
        return BBSum(self, other)

    def sigmoid(self) -> BBSigmoid:
        return BBSigmoid(self)

    def mse(self, y: Any) -> BBMSELoss:
        return BBMSELoss(self, y)

    def bce(self, y: Any) -> BBBCELoss:
        return BBBCELoss(self, y)

    def save(self) -> str:
        return json.dumps(self.to_json())

    def to_json(self) -> Any:
        results = []
        for arg in self.args:
            if isinstance(arg, list):
                # Reduce precision to compress the model and bring a bit of noise
                rounded = [[round(x, 5) for x in row] for row in arg]
                # TODO: maybe replace args with new values?
                results.append(rounded)
            elif isinstance(arg, BB):
                results.append(arg.to_json())
            else:
                raise ValueError("Unserializable arg: ", type(arg).__name__)
        return results

    def load(self, jsonx: str) -> None:
        return self.from_json(json.loads(jsonx))

    def from_json(self, jsoned: Any) -> None:
        for i, res in enumerate(jsoned):
            if isinstance(self.args[i], list):
                self.args[i] = res
            elif isinstance(self.args[i], BB):
                self.args[i].from_json(res)
            else:
                raise ValueError(
                    f"Trouble deserializing arg #{i}: ", type(self.args[i]).__name__
                )


class BBSum(BB):
    def __init__(self, arg1: BB, arg2: BB) -> None:
        super().__init__(arg1, arg2)

        if self.arg(0).dims() != self.arg(1).dims():
            raise ValueError(
                f"Dimentions unmatch input->{self.arg(0).dims()}, bias->{self.arg(1).dims()}"
            )

    def val(self) -> Any:
        self.value = add(self.arg(0).val(), self.arg(1).val())
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        super().dif(dvalue)
        self.arg(0).dif(self.dval())
        self.arg(1).dif(self.dval())


def reshape(matrix: list[list[float]], o: int, p: int) -> list[list[float]]:
    flat_list = sum(matrix, [])  # Flatten the matrix
    return [flat_list[i * p : (i + 1) * p] for i in range(o)]


class BBReshape(BB):
    m: int
    n: int
    o: int
    p: int
    def __init__(self, arg: BB, o: int, p: int) -> None:
        super().__init__(arg)

        m, n = self.arg(0).dims()
        if m * n != o * p:
            raise ValueError(f"Incompatible reshape {m} * {n} != {o} * {p}")
        self.m, self.n = m, n
        self.o, self.p = o, p

    def val(self) -> Any:
        self.value = reshape(self.arg(0).val(), self.o, self.p)
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        super().dif(dvalue)
        self.arg(0).dif(reshape(dvalue, self.m, self.n))


class BBMatmul(BB):
    input: BB
    weights: BB
    def __init__(self, arg1: BB, arg2: BB) -> None:
        super().__init__(arg1, arg2)
        self.input = self.arg(0)
        self.weights = self.arg(1)

        if self.input.dims()[1] != self.weights.dims()[0]:
            raise ValueError(
                f"Dimentions unmatch input->{self.input.dims()}, weights->{self.weights.dims()}"
            )

    def val(self) -> Any:
        self.value = matmul(self.input.val(), self.weights.val())
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        super().dif(dvalue)
        self.weights.dif(matmul(transpose(self.input.val()), self.dval()))
        self.input.dif(matmul(self.dval(), transpose(self.weights.val())))


class BBSigmoid(BB):
    inp: BB
    def __init__(self, inp: BB) -> None:
        super().__init__(inp)
        self.inp = inp

    def val(self) -> Any:
        self.value = vectorized(sigmoid, self.inp.val())
        return self.value

    def derivative(self) -> Any:
        return sigmoid_derivative(self.inp.val())

    def dif(self, dvalue: Any = None) -> None:
        self.dvalue = elementwise_mul(dvalue, self.derivative())
        self.inp.dif(self.dval())


# Mean Squared Error
class BBMSELoss(BB):
    inp: BB
    y: BB
    def __init__(self, inp: BB, y: BB) -> None:
        super().__init__(inp, y)
        self.inp = inp
        self.y = y

    # TODO: actually derivative must be 0.5 * ..., but TODO
    def derivative(self) -> Any:
        return subtract(self.inp.val(), self.y.val())  # derivative of loss

    def val(self) -> Any:
        self.value = [
            [(a - b) ** 2 for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.inp.val(), self.y.val())
        ]
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        self.dvalue = self.derivative()
        self.inp.dif(self.dval())


# Binary cross enthropy
class BBBCELoss(BB):
    inp: BB
    y: BB
    def __init__(self, inp: BB, y: BB) -> None:
        super().__init__(inp, y)
        self.inp = inp
        self.y = y

    def derivative(self) -> Any:
        return self.bce_derivative()
        # return subtract(self.inp.val(), self.y.val()) # derivative of loss

    def bce_derivative(self) -> Any:
        # Clamp p to avoid division by zero
        epsilon = 1e-12
        dval = []
        for row_a, row_b in zip(self.y.val(), self.inp.val()):
            for y, p in zip(row_a, row_b):
                p = max(min(p, 1 - epsilon), epsilon)
                dval.append((p - y) / (p * (1 - p)))
        return [dval]

    def val(self) -> Any:
        loss = []
        eps = 1e-15
        for row_a, row_b in zip(self.y.val(), self.inp.val()):
            for y, p in zip(row_a, row_b):
                # Clip p to avoid log(0)
                p = min(max(p, eps), 1 - eps)
                l = -(y * math.log(p) + (1 - y) * math.log(1 - p))
                loss.append(l)
        self.value = [loss]
        return self.value

    def dif(self, dvalue: Any = None) -> None:
        self.dvalue = self.derivative()
        self.inp.dif(self.dval())
