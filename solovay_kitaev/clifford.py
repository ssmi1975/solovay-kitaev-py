import functools
import collections

from .algorithm import Uop, GateSet

SQRT2 = 2 ** 0.5
ALPHA = (2 + SQRT2) ** 0.5 / 2
BETA = (2 - SQRT2) ** 0.5 / 2

def clifford_set(u):
    """ create length24 list of [(u*C) for C in Clifford]
    Clifford can be constructed with product((I,iX,I+iX,I-iX,I+iY,I-iY) , (I,iZ,I+iZ,I-iZ))"""
    i, x, y, z = u.v
    result = []
    result.append(u.clone()) # I
    result.append(Uop(-x, i, -z, y, u.hierarchy, u.construction + ["X"], gateset=u.gateset)) # iX, but treat it as X due to only phase difference
    result.append(Uop((i-x)/SQRT2, (x+i)/SQRT2, (y-z)/SQRT2, (z+y)/SQRT2, u.hierarchy, u.construction + ["(I+iX)"], gateset=u.gateset))
    result.append(Uop((i+x)/SQRT2, (x-i)/SQRT2, (y+z)/SQRT2, (z-y)/SQRT2, u.hierarchy, u.construction + ["(I-iX)"], gateset=u.gateset))
    result.append(Uop((i-y)/SQRT2, (x+z)/SQRT2, (y+i)/SQRT2, (z-x)/SQRT2, u.hierarchy, u.construction + ["(I+iY)"], gateset=u.gateset))
    result.append(Uop((i+y)/SQRT2, (x-z)/SQRT2, (y-i)/SQRT2, (z+x)/SQRT2, u.hierarchy, u.construction + ["(I-iY)"], gateset=u.gateset))
    for idx in range(6):
        i, x, y, z = result[idx].v
        c = result[idx].construction[-1:] if idx != 0 else []
        result.append(Uop(-z, -y, x, i, u.hierarchy, u.construction + c + ["Z"], gateset=u.gateset)) # iZ
        result.append(Uop((i-z)/SQRT2, (x-y)/SQRT2, (y+x)/SQRT2, (z+i)/SQRT2, u.hierarchy, u.construction + c + ["(I+iZ)"], gateset=u.gateset))
        result.append(Uop((i+z)/SQRT2, (x+y)/SQRT2, (y-x)/SQRT2, (z-i)/SQRT2, u.hierarchy, u.construction + c + ["(I-iZ)"], gateset=u.gateset))

    return result

CLIFFORD_DAGGERS = {
    "I": ("I",),
    "X": ("X",),
    "Y": ("Y",),
    "Z": ("Z",),
    "(I+iX)": ("(I-iX)",),
    "(I-iX)": ("(I+iX)",),
    "(I+iY)": ("(I-iY)",),
    "(I-iY)": ("(I+iY)",),
    "(I+iZ)": ("(I-iZ)",),
    "(I-iZ)": ("(I+iZ)",),
    "T": ("X", "T", "X"),
    "H": ("H",),
}

def generate_epsilon_network(gateset, max_hierarchy=3, ordering=True):
    I = Uop(1, 0, 0, 0, gateset=gateset)
    result = clifford_set(I)
    queue = collections.deque(result)
    current_hierarchy = 0

    while len(queue) > 0:
        u = queue.popleft()
        i, x, y, z = u.v
        ti = i*ALPHA - z*BETA
        tx = x*ALPHA - y*BETA
        ty = y*ALPHA + x*BETA
        tz = z*ALPHA + i*BETA

        circuits = clifford_set(Uop(ti, tx, ty, tz, u.hierarchy + 1, u.construction + ["T"], gateset=gateset))
        if u.hierarchy + 1 > current_hierarchy:
            current_hierarchy += 1

        for c in circuits:
            if any((c.is_similar(o) for o in result)):
                continue
            result.append(c)
            if c.hierarchy < max_hierarchy:
                queue.append(c)
    
    if ordering:
        # keep the same order as the original C++ code
        from operator import attrgetter
        result = sorted(result, key=attrgetter('i', 'x', 'y', 'z'))

    return result


def clifford_gateset(depth=3):
    return GateSet(
        lambda self, gate: gate in ("I", "X", "(I+iX)", "(I-iX)",
                              "(I+iY)", "(I-iY)", "Z", "(I+iZ)", "(I-iZ)", "T", "H"),
        lambda self, gate: CLIFFORD_DAGGERS[gate],
        functools.partial(generate_epsilon_network, max_hierarchy=depth)
    )

