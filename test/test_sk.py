from solovay_kitaev import solovay_kitaev as sk
import math
import pytest
from math import sin, cos, pi, e


#@pytest.mark.parametrize("count, expected",
        #[1, "(I-iZ)T"])
def test():
    #t = sk.Uop(0, 1, 0, 0, 0, [])
    t = sk.Uop(math.cos(math.pi / 16), 0, 0, math.sin(math.pi / 16))
    print(t)
    s = sk.generate_epsilon_network()
    for i in range(5):
        print(i)
        result = sk.solovay_kitaev(s, t, i)
        print(result.construction_str())
        print("Uop: {}".format(result))
        print("diff: {}".format(result.operator_distance(t)))
    assert False

ERR = 10**-5
SQRT2 = sk.SQRT2

@pytest.mark.parametrize("matrix, expected",[
        ([[1,0],[0,1]], (1, 0, 0, 0)),  # I
        ([[0,1],[1,0]], (0, 1, 0, 0)),  # X
        ([[0,-1j],[1j,0]], (0, 0, 1, 0)),  # Y
        ([[1,0],[0,-1]], (0, 0, 0, 1)),  # Z
        ([[1,1],[1,-1]], (0, 1/SQRT2, 0, 1/SQRT2)),  # Hadamard
])
def test_from_matrix(matrix, expected):
    # Hadamard -> 
    u = sk.Uop.from_matrix(matrix)
    for i in range(4):
        assert expected[i] - ERR < u.v[i] < expected[i] + ERR

@pytest.mark.parametrize("args",[
        (1, 0, 0, 0),  # I
        (0, 1, 0, 0),  # X
        (0, 0, 1, 0),  # Y
        (0, 0, 0, 1),  # Z
        (0, 1/SQRT2, 0, 1/SQRT2),  # Hadamard
])
def test_matrix_form(args):
    # Hadamard -> 
    u = sk.Uop(*args)
    assert u == sk.Uop.from_matrix(u.matrix_form())


if __name__ == "__main__":
    test()
