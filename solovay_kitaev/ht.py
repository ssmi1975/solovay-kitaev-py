import functools
import math

from .algorithm import Uop, GateSet



def generate_epsilon_network(gateset, length=17):
    H = Uop.from_matrix([[1, 1],[1, -1]], 0, ['H'], True, gateset)
    T = Uop(math.cos(math.pi/8.), 0, 0, math.sin(math.pi/8.), 0, ['T'], gateset=gateset)
    result = []
    #result.append(Uop(1, 0, 0, 0, gateset=gateset))
    result.append(H)
    result.append(T)
    for i in range(2, length + 1):
        new_results = []
        for u in result:
            if len(u.construction) < i - 1:
                continue
            if u.construction[-1] == 'H':
                # avoid H*H = I
                new_results.append(u @ T)
            else:
                new_results.append(u @ H)
                if not (len(u.construction) >= 7 and u.construction[-7:] == [T] * 7):
                    # avoid T ** 8 = I
                    new_results.append(u @ T)
        for n in new_results:
            if any(n.is_similar(u) for u in result):
                continue
            result.append(n)
    return result


DAGGERS = {
    "H": ("H",),
    "T": ("T", "T", "T", "T", "T", "T", "T"),
}


def simplify(construction):
    # devise the fact that there's only single-char gates
    s = "".join(construction)
    while True:
        l0 = len(s)
        s = s.replace('HH', '').replace('TTTTTTTT', '')
        l1 = len(s)
        if l0 == l1:
            break
    return list(s)


def ht_gateset(length=17):
    return GateSet(
        lambda self, gate: gate in ("H", "T"),
        lambda self, gate: DAGGERS[gate],
        functools.partial(generate_epsilon_network, length=length),
        lambda self, constructions: simplify(constructions)
    )

