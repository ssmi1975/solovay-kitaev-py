from solovay_kitaev import solovay_kitaev as sk
import math
import pytest
from math import sin, cos, pi, e

Uop = sk.Uop
SQRT2 = sk.SQRT2
ERR = 10**-5

oI = Uop(1, 0, 0, 0, 0)
oX = Uop(0, 1, 0, 0, 0, ['X'])
osX = Uop(1 / SQRT2, 1 / SQRT2, 0, 0, 0, ['(I+iX)'])
osXd = Uop(1 / SQRT2, -1 / SQRT2, 0, 0, 0, ['(I-iX)'])
osY = Uop(1 / SQRT2, 0, 1 / SQRT2, 0, 0, ['(I+iY)'])
osYd = Uop(1 / SQRT2, 0, -1 / SQRT2, 0, 0, ['(I-iY)'])
oZ = Uop(0, 0, 0, 1, 0, ['Z'])
osZ = Uop(1 / SQRT2, 0, 0, 1 / SQRT2, 0, ['(I+iZ)'])
osZd = Uop(1 / SQRT2, 0, 0, -1 / SQRT2, 0, ['(I-iZ)'])
T = Uop(cos(pi/8.), 0, 0, sin(pi/8.), 0, ['T'])
gates = {"I": oI,
        "X": oX,
        "(I+iX)": osX,
        "(I-iX)": osXd,
        "(I+iY)": osY,
        "(I-iY)": osYd,
        "Z": oZ,
        "(I+iZ)": osZ,
        "(I-iZ)": osZd,
        "T": T}


@pytest.mark.parametrize("left, right, expected",(
        [oI, oZ, SQRT2],
        [oZ, oI, SQRT2],
        [oI, osX, 1],
        [oZ, osX, math.sqrt(3)],
        [oZ, oZ, 0],
))
def test_distance(left, right, expected):
    assert expected - ERR < left.operator_distance(right) < expected + ERR
    
def verify_construction(u):
    from_construction = []
    from_construction.append(oI)
    for c in u.construction:
        op = from_construction[-1] @ gates[c]
        from_construction.append(op)
    assert u.construction == from_construction[-1].construction
    assert u.is_similar(from_construction[-1]), "approximate: {}, from_construction: {}".format(u, from_construction[-1])

@pytest.mark.parametrize("count, expected",(
        [0, "T"],
        [1, "(I+iY)T(I-iY)(I+iX)T(I-iY)(I-iZ)(I+iY)XTX(I-iY)(I+iZ)(I+iY)XTX(I-iX)T"],
))
def test(count, expected):
    t = sk.Uop(math.cos(math.pi / 16), 0, 0, math.sin(math.pi / 16))
    print(t)
    s = sk.generate_epsilon_network(3)
    result = sk.solovay_kitaev(s, t, count)
    print(result)
    print("dist: {}".format(result.operator_distance(t)))
    assert expected == result.construction_str()


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
