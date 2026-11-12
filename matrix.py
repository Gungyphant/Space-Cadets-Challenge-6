import math
from functools import cache
from typing import overload

import numpy as np
from PIL import Image as ImageModule
from PIL.Image import Image


def clamp(lower, value, upper):
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


class Matrix:
    values: list[list[int]]
    width: int
    height: int

    @overload
    def __init__(self, values: list[list[int]]):
        """Initialise a Matrix with the provided values

        :param values: A list of lists of ints, where each list represents a
                       row of values in the matrix
        """
        pass

    @overload
    def __init__(self, width: int, height: int):
        """Initialise an empty Matrix with specified dimensions

        :param width: The width of the empty Matrix
        :param height: The height of the empty Matrix
        """
        pass

    @overload
    def __init__(self, image: Image):
        """Initialise a Matrix with the pixel values of a Pillow Image

        :param image: The Image to get data from
        """
        pass

    def __init__(self, *args):
        match len(args):
            case 1:
                arg = args[0]
                if isinstance(arg, list):
                    values = arg

                    self.values = values
                    self.width = len(values[0])
                    self.height = len(values)

                elif isinstance(arg, Image):
                    image = arg

                    # noinspection PyTypeChecker
                    image_data = list(image.getdata())
                    self.width = image.width
                    self.height = image.height
                    self.values = [
                        [int(n) for n in image_data[i:i+self.width]]
                        for i in range(0, len(image_data), self.width)
                    ]


            case 2:
                width, height = args

                self.values = [
                    [0 for _ in range(width)]
                    for _ in range(height)
                ]
                self.width = width
                self.height = height

            case _:
                raise TypeError(f"Matrix.__init__() takes 1 or 2 positional "
                                f"arguments but {len(args)} were given")

    def __getitem__(self, item):
        return self.values[item]

    @property
    @cache
    def center(self) -> tuple[int, int]:
        return self.width // 2, self.height // 2

    def __mul__(kernel, source: Matrix) -> Matrix:
        # Note: this is convolution, not multiplication (both are signified by
        #  *)
        if not isinstance(source, Matrix):
            raise TypeError(f"Can't convolute a matrix by a '{type(kernel)}'")

        kernel_center_x, kernel_center_y = kernel.center
        result = Matrix(source.width, source.height)
        for source_x in range(source.width):
            for source_y in range(source.height):
                value = 0
                for kernel_x in range(kernel.width):
                    corresponding_source_x = source_x + (kernel_x - kernel_center_x)

                    # Use the extend edge-handling method:
                    corresponding_source_x = clamp(0, corresponding_source_x, source.width - 1)
                    for kernel_y in range(kernel.height):
                        corresponding_source_y = source_y + (kernel_y - kernel_center_y)

                        # Use the extend edge-handling method:
                        corresponding_source_y = clamp(0, corresponding_source_y, source.height - 1)

                        corresponding_source_value = source[corresponding_source_y][corresponding_source_x]
                        kernel_value = kernel[kernel_y][kernel_x]

                        value += corresponding_source_value * kernel_value

                result[source_y][source_x] = value

        return result

    def __str__(self):
        return (f"⌈{", ".join(f"{n: >3d}" for n in self[0])}⌉\n"
                f"{"\n".join(f"│{", ".join(f"{n: >3d}" for n in line)}│" for line in self[1:-1])}\n"
                f"⌊{", ".join(f"{n: >3d}" for n in self[-1])}⌋")

    def __repr__(self):
        return repr(self.values)

    def __pow__(self, power, modulo=None) -> Matrix:
        return Matrix([[n ** 2 for n in line] for line in self])

    def __add__(self, other: Matrix) -> Matrix:
        if not (self.width == other.width and self.height == other.height):
            raise ValueError(f"Incompatible dimensions for addition")
        return Matrix([
            [n + other[y][x] for x, n in enumerate(line)]
            for y, line in enumerate(self.values)]
        )

    def sqrt(self) -> Matrix:
        return Matrix([[int(math.sqrt(n)) for n in line] for line in self])

    def max(self) -> int:
        return max(*[max(*line) for line in self])

    def min(self) -> int:
        return min(*[min(*line) for line in self])

    def normalise(self, max_valid: int = 1, min_valid:int = 0) -> Matrix:
        if max_valid == 1 and min_valid == 0:
            max_val = self.max()
            min_val = self.min()
            return Matrix([
                [(n - min_val)/(max_val - min_val) for n in line]
                for line in self
            ])
        else:
            if min_valid == 0:
                return Matrix([
                    [int(n * max_valid) for n in line]
                    for line in self.normalise()
                ])
            else:
                return Matrix([
                    [n + min_valid for n in line]
                    for line in self.normalise(max_valid)
                ])

    def to_image(self) -> Image:
        return ImageModule.fromarray(
            np.asarray(self.values, dtype=np.uint8),
            "L"
        )

    def max_and_indices(self) -> tuple[int, int, int]:
        max_found = -1
        max_found_x = -1
        max_found_y = -1
        for y, line in enumerate(self.values):
            for x, val in enumerate(line):
                if val > max_found:
                    max_found = val
                    max_found_x = x
                    max_found_y = y
        return max_found, max_found_x, max_found_y


if __name__ == "__main__":
    # Example from https://en.wikipedia.org/wiki/Kernel_(image_processing)#/media/File:2D_Convolution_Animation.gif
    kernel = Matrix([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ])

    source = Matrix([
        [7, 6, 5, 5, 6, 7],
        [6, 4, 3, 3, 4, 6],
        [5, 3, 2, 2, 3, 5],
        [5, 3, 2, 2, 3, 5],
        [6, 4, 3, 3, 4, 6],
        [7, 6, 5, 5, 6, 7],
    ])

    print(kernel * source)
