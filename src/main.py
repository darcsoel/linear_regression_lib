"""
Linear regression implelemtation.
File contains model for train data and
generator of training data.

"""

from random import randint
from typing import Self


def generate_test_data(length: int = 100) -> tuple[list[int], list[int]]:
    x: list[int] = []
    y: list[int] = []

    for i in range(length):
        x.append(i)
        y.append((i * randint(1, 5)) // 2)

    return x, y


class LinearRergession:
    """
    Linear regression is represented by formula:
        y = mx + b

    """

    def __init__(self, x: list[int], y: list[int]) -> None:
        self._x = x
        self._y = y

        self._m: float | None = None
        self._b: float | None = None

    def _find_m(self) -> Self:
        s = 0
        for x, y in zip(self._x, self._y, strict=True):
            s += x * y

        n = len(self._x)

        upper: int = s * n - sum(self._x) * sum(self._y)
        lower: int = n * sum(x**2 for x in self._x) - sum(self._x) ** 2

        self._m = upper / lower
        return self

    def _find_b(self) -> Self:
        if not self._m:
            raise RuntimeError("Value of M was not calculated. Please call methods in correct order.")

        self._b = (sum(self._y) - self._m * sum(self._x)) / len(self._x)
        return self

    def train(self) -> tuple[float, float]:
        self._find_m()
        self._find_b()

        if self._m is None or self._b is None:
            raise RuntimeError("Model was not trained. Aborting.")

        return self._m, self._b

    def get_coeficients(self) -> tuple[float | None, float | None]:
        return self._m, self._b


def predict(x: int, m: float, b: float) -> int | float:
    return x * m + b


if __name__ == "__main__":
    data: tuple[list[int], list[int]] = generate_test_data(1000)

    model = LinearRergession(data[0], data[1])
    m, b = model.train()

    print(data[0][-1])
    print(data[1][-1])

    predicted = predict(1001, m, b)

    print(f"Model trained parameters: {model.get_coeficients()}")
    print(predicted)
