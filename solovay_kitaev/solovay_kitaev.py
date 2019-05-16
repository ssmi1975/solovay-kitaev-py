import numpy as np
import math
import copy
import collections

# direct translation from https://github.com/kodack64/SolovayKitaevAlg

EPS = 1e-12
SQRT2 = 2 ** 0.5
ALPHA = (2 + SQRT2) ** 0.5 / 2
BETA = (2 - SQRT2) ** 0.5 / 2
NUM_CLIFFORD = 24
GATESTR = ["I","X","(I+iX)","(I-iX)","(I+iY)","(I-iY)","Z","(I+iZ)","(I-iZ)","T"]
DAG_ID = [ 0, 1, 3, 2, 5, 4, 6, 8, 7, -1 ]
TID = 9

# allowed number of T-gate in epsilon net
MAX_HIERARCHY = 3

# maximum recursive number of Solovay-Kitaev
MAX_RECURSIVE_SK = 10

# If ture, normalize when unitary operator is defined
NORMALIZE = True

# If ture, print construction of gates with Clifford gates and T-gates.
SHOW_CONSTRUCTION = False

class Uop:
    def __init__(self, i, x, y, z, hierarchy, construction, buildFlag=True, normalize=True):
        self.hierarchy = hierarchy
        self.construction = construction
        self.buildFlag = buildFlag
        self.v = (i, x, y, z)
        self._fixup_direction()
        if normalize:
            norm = sum([v ** 2 for v in self.v]) ** 0.5
            self.v = tuple((v / norm for v in self.v))
        else:
            # assert norm equals 1.
            assert 1 -EPS < sum((v**2 for v in self.v)) < 1 + EPS
        assert type(self.construction) == list
        for c in self.construction:
            assert type(c) == int, "bad construction: %s" % self.construction

    def _fixup_direction(self):
        """to omit double-counting in SU(2), every variable is inverted if the first non-zero value is negative"""
        for v in self.v:
            if v < EPS:
                continue
            if v < 0:
                self.v = tuple((-value for value in self.v))
                break

    @property
    def i(self):
        return self.v[0]

    @property
    def x(self):
        return self.v[1]

    @property
    def y(self):
        return self.v[2]

    @property
    def z(self):
        return self.v[3]

    def clone(self):
        return copy.deepcopy(self)

    def matrix_form(self):
        i, x, y, z = self.v
        return np.array([[i+z*1j, y+x*1j], [-y+x*1j, i-z*1j]])

    def __matmul__(self, other):
        # see Nielsen & Chuang, Exercise 4.15 (1)
        i1, x1, y1, z1 = self.v
        i2, x2, y2, z2 = other.v
        ni = i1*i2 - x1*x2 - y1*y2 - z1*z2
        nx = i1*x2 + x1*i2 - y1*z2 + z1*y2
        ny = i1*y2 + x1*z2 + y1*i2 - z1*x2
        nz = i1*z2 - x1*y2 + y1*x2 + z1*i2
        return Uop(ni, nx, ny, nz, self.hierarchy + other.hierarchy, self.construction + other.construction, self.buildFlag & other.buildFlag)

    def dagger(self):
        ncon = []
        for c in self.construction:
            if c == TID:
                # T^dag = XTX
                ncon += [1, TID, 1]
            else:
                ncon.append(DAG_ID[c])
        return Uop(self.i, -self.x, -self.y, -self.z, self.hierarchy, ncon, self.buildFlag)

    def __str__(self):
        #nn = sum((value ** 0.5 for value in self.v[1:]))
        return """ ****
#T gate : {self.hierarchy}
{self.v[0]} I + {self.v[1]} iX + {self.v[2]} iY + {self.v[3]} iZ
rot = {rot}
 **** """.format(
        self=self,
        rot= 2 * math.acos(self.v[0])/math.pi
    )

    def set_clifford(self):
        """ create length24 list of [(u*C) for C in Clifford]
        Clifford can be constructed with product((I,iX,I+iX,I-iX,I+iY,I-iY) , (I,iZ,I+iZ,I-iZ))"""
        i, x, y, z = self.v
        result = []
        result.append(self.clone()) # I
        result.append(Uop(-x, i, -z, y, self.hierarchy, self.construction + [1])) # iX
        result.append(Uop((i-x)/SQRT2, (x+i)/SQRT2, (y-z)/SQRT2, (z+y)/SQRT2, self.hierarchy, self.construction + [2])) # I+iX
        result.append(Uop((i+x)/SQRT2, (x-i)/SQRT2, (y+z)/SQRT2, (z-y)/SQRT2, self.hierarchy, self.construction + [3])) # I-iX
        result.append(Uop((i-y)/SQRT2, (x+z)/SQRT2, (y+i)/SQRT2, (z-x)/SQRT2, self.hierarchy, self.construction + [4])) # I+iY
        result.append(Uop((i+y)/SQRT2, (x-z)/SQRT2, (y-i)/SQRT2, (z+x)/SQRT2, self.hierarchy, self.construction + [5])) # I-iY
        for idx in range(6):
            i, x, y, z = result[idx].v
            c = [idx] if idx != 0 else []
            result.append(Uop(-z, -y, x, i, self.hierarchy, self.construction + c + [6]))
            result.append(Uop((i-z)/SQRT2, (x-y)/SQRT2, (y+x)/SQRT2, (z+i)/SQRT2, self.hierarchy, self.construction + c + [7]))
            result.append(Uop((i+z)/SQRT2, (x+y)/SQRT2, (y-x)/SQRT2, (z-i)/SQRT2, self.hierarchy, self.construction + c + [8]))

        return result

    def __hash__(self):
        return hash((self.v, self.hierarchy, tuple(self.construction)))
        
    def construction_str(self):
        return "".join([GATESTR[c] for c in self.construction])

    def operator_distance(self, other):
        u = self.matrix_form()
        v = other.matrix_form()
        values = np.linalg.eigvalsh(u-v)
        #print(values)
        return values[-1] ** 0.5

    def get_similar(self, uop_list):
        result = None
        distance = math.inf
        for u in uop_list:
            dist = self.operator_distance(u)
            if distance > dist:
                distance = dist
                result = copy.deepcopy(u)
        return result


I = Uop(1, 0, 0, 0, 0, [])


def generate_epsilon_network():
    result = set(I.set_clifford())
    queue = collections.deque(result)
    current_hierarchy = 0

    while len(queue) > 0:
        u = queue.popleft()
        i, x, y, z = u.v
        ti = i*ALPHA - z*BETA
        tx = x*ALPHA - y*BETA
        ty = y*ALPHA + x*BETA
        tz = z*ALPHA + i*BETA

        circuits = Uop(ti, tx, ty, tz, u.hierarchy + 1, u.construction + [TID]).set_clifford()
        if u.hierarchy + 1 > current_hierarchy:
            current_hierarchy += 1

        for c in circuits:
            if c in result:
                break
            if c.hierarchy < MAX_HIERARCHY:
                result.add(c)
    return result


def gc_decompose(udd):
    # udd = Rn(θ)
    s = ((1 - udd.i) / 2) ** 0.25   # sin Φ = ((1 - cos θ) / 2) ** (1/4)
    c = (1 - s ** 2) ** 0.5         # cos Φ  = 1 - sin Φ
    v = Uop(c, s, 0, 0, 0, [], False)   # v = cosΦ I - i sinΦ X = Rx(Φ)
    w = Uop(c, 0, s, 0, 0, [], False)   # v = cosΦ I - i sinΦ Y = Ry(Φ)

    # n = (nx, ny, nz)
    nn = (1 - udd.i ** 2) ** 0.5
    nx = udd.x / nn
    ny = udd.y / nn
    nz = udd.z / nn

    mn = nn # ??
    mx = 2 * s ** 3 * c / mn    # mx = 2 * sinΦ ** 3 * cosΦ
    my = -2 * s ** 3 * c / mn
    mz = -2 * s ** 2 * c ** 2 / mn

    x = (nx + mx) / 2
    y = (ny + my) / 2
    z = (nz + mz) / 2
    n = (x*x + y*y + z*z) ** 0.5
    x /= n
    y /= n
    z /= n

    s = Uop(0, x, y, z, 0, [], False)
    vt = s @ v @ s.dagger()
    wt = s @ w @ s.dagger()

    return vt, wt



def solovay_kitaev(uop_set, u, rec):
    """
    SK

    for given u, rec

    ud = SK(u,rec-1)
    udd = u ud
    decompose udd = S V W V.dag W.dag S.dag
        where    
                udd    = a I - i b X - i c Y - i d Z
                a    = cos(theta/2)
                V    = cos(phi/2) I - i sin(phi/2) X
                W    = cos(phi/2) I - i sin(phi/2) Y

    V W V.dag W.dag = (1-2s^4, 2s^3c,-2s^3c, -2c^2s^2)

    a = 1-2s^4 <==>    s = pow((1-a)/2,1/4)

    Vt = S V S.dag
    Wt = S W S.dag

    udd = Vt Wt Vt.dag Wt.dag
    Vd = SK(Vt,rec-1)
    Wd = SK(Wt,rec-1)

    return Vd Wd Vd.ag Wd.dag ud
    """

    if rec == 0:
        return u.get_similar(uop_set)
    ud = solovay_kitaev(uop_set, u, rec - 1)
    udd = u @ ud.dagger()

    vt, wt = gc_decompose(udd)

    vd = solovay_kitaev(uop_set, vt, rec - 1)
    wd = solovay_kitaev(uop_set, wt, rec - 1)
    return vd @ wd @ vd.dagger() @ wd.dagger() @ ud
