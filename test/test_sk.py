from solovay_kitaev import execute_solovay_kitaev, Uop
import solovay_kitaev.algorithm as sk
import solovay_kitaev.clifford as cl
import math
import pytest
from math import sin, cos, pi, e

SQRT2 = math.sqrt(2)
ERR = 10**-12

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
H = Uop.from_matrix([[1, 1],[1, -1]], 0, ['H'])
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

def check_sk(target, count, expected):
    print(target)
    result = execute_solovay_kitaev(target, count)
    print(result)
    print("dist: {}".format(result.operator_distance(target)))
    assert expected == result.construction_str()

@pytest.mark.parametrize("count, expected",(
        [0, "T"],
        [1, "(I+iY)T(I-iY)(I+iX)T(I-iY)(I-iZ)(I+iY)XTX(I-iY)(I+iZ)(I+iY)XTX(I-iX)T"],
        [2, "T(I+iX)T(I+iX)TXT(I+iY)T(I+iX)TZXXTX(I-iX)XTX(I-iX)XTXZXTX(I-iX)XTX(I-iY)XTX(I+iY)T(I+iX)T(I+iY)T(I+iX)(I+iZ)(I+iY)T(I-iY)(I+iY)T(I+iY)T(I+iX)T(I+iY)Z(I+iY)XTX(I-iY)Z(I-iY)XTX(I-iX)XTX(I-iY)XTX(I-iY)(I+iX)T(I+iX)T(I+iY)T(I+iY)(I-iZ)(I-iZ)(I-iX)XTX(I-iY)XTX(I-iX)XTX(I-iY)XXTXX(I+iY)XXTXX(I+iX)XXTXXZXXTXX(I+iX)XXTXX(I+iX)XXTXXXZXTX(I-iX)XTX(I-iY)XTXXXTX(I-iX)XTX(I-iX)XTX(I+iZ)(I-iY)XTX(I-iY)XTX(I-iX)XTX(I-iX)(I+iY)XXTXX(I+iY)XXTXX(I+iX)XXTXX(I+iY)Z(I+iY)XXTXX(I-iY)Z(I-iY)XTX(I-iX)XTX(I-iY)XTX(I-iY)(I+iY)XTX(I-iY)(I+iY)T(I-iY)(I+iX)T(I-iY)(I-iZ)(I+iY)XTX(I-iY)(I+iZ)(I+iY)XTX(I-iX)T"],
))
def test(count, expected):
    t = sk.Uop(math.cos(math.pi / 16), 0, 0, math.sin(math.pi / 16), gateset=cl.clifford_gateset())
    check_sk(t, count, expected)

@pytest.mark.parametrize("count, expected",(
        [0, "T"],
        [1, "(I+iY)T(I-iY)(I+iX)T(I-iY)(I-iZ)(I+iY)XTX(I-iY)(I+iZ)(I+iY)XTX(I-iX)T"],
        [2, "T(I+iX)T(I+iX)TXT(I+iY)T(I+iX)TZXXTX(I-iX)XTX(I-iX)XTXZXTX(I-iX)XTX(I-iY)XTX(I+iY)T(I+iX)T(I+iY)T(I+iX)(I+iZ)(I+iY)T(I-iY)(I+iY)T(I+iY)T(I+iX)T(I+iY)Z(I+iY)XTX(I-iY)Z(I-iY)XTX(I-iX)XTX(I-iY)XTX(I-iY)(I+iX)T(I+iX)T(I+iY)T(I+iY)(I-iZ)(I-iZ)(I-iX)XTX(I-iY)XTX(I-iX)XTX(I-iY)XXTXX(I+iY)XXTXX(I+iX)XXTXXZXXTXX(I+iX)XXTXX(I+iX)XXTXXXZXTX(I-iX)XTX(I-iY)XTXXXTX(I-iX)XTX(I-iX)XTX(I+iZ)(I-iY)XTX(I-iY)XTX(I-iX)XTX(I-iX)(I+iY)XXTXX(I+iY)XXTXX(I+iX)XXTXX(I+iY)Z(I+iY)XXTXX(I-iY)Z(I-iY)XTX(I-iX)XTX(I-iY)XTX(I-iY)(I+iY)XTX(I-iY)(I+iY)T(I-iY)(I+iX)T(I-iY)(I-iZ)(I+iY)XTX(I-iY)(I+iZ)(I+iY)XTX(I-iX)T"],
))
def test_from_matrix():
    t = sk.Uop.from_matrix([[1, 0],[0, math.e ** (math.pi / 8 + 1j)]])
    check_sk(t, count, expected)


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


@pytest.mark.skip()
@pytest.mark.parametrize("iteration,expected",(
    [1, [oI, H, T]],
    [2, [oI, H, T, H@T, T@H, T@T]],
    [3, [oI, H, T, H@T, T@H, T@T, H@T@H, H@T@T, T@H@T, T@T@H, T@T@T]],
))
def test_epsilon_ht(iteration, expected):
    assert expected == sk.generate_epsilon_network_ht(iteration)

@pytest.mark.skip()
def test_sk_ht():
    target = sk.Uop(math.cos(math.pi / 16), 0, 0, math.sin(math.pi / 16), cl.clifford_gateset())
    s = sk.generate_epsilon_network_ht(16)
    import pprint
    pprint.pprint(s)
    for i in range(5):
        result = execute_solovay_kitaev(target, i)
        print(f"\niteration: {i}")
        print(result)
        print(f"dist: {result.operator_distance(target)}")
        print(f"gates: {len(result.construction)}")
        print(f"gates: {result.construction_str()}")
    assert False


if __name__ == "__main__":
    test()
