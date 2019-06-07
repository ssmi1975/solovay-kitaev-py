import math
import copy
import collections
import functools

# most of the code are translated from https://github.com/kodack64/SolovayKitaevAlg

EPS = 1e-12

# XXX : any better way to impelement strategy pattern?
class GateSet:
    def __init__(self, valid_construction, get_dagger, epsilon_net):
        self._valid_construction = valid_construction
        self._get_dagger = get_dagger
        self._epsilon_net = epsilon_net

    def is_valid_construction(self, gate):
        return self._valid_construction(self, gate)

    def get_dagger(self, gate):
        return self._get_dagger(self, gate)

    def epsilon_net(self):
        return self._epsilon_net(self)


NULL_GATESET = GateSet(
    lambda self, gate: True,
    lambda self, gate: "",
    lambda self: []
)


class Uop:
    def __init__(self, i, x, y, z, hierarchy=0, construction=[], normalize=True, gateset=NULL_GATESET):
        # U = iI + jxX + jyY + jzZ
        #   = 0.5 * (cos(Φ/2)*I + j*sin(Φ/2)*(x*X + y*Y + z*Z) )
        self.hierarchy = hierarchy
        self.construction = construction
        self.v = (i, x, y, z)
        self.gateset = gateset
        if self._needs_fixup_direction():
            self.v = tuple((-value for value in self.v))
        if normalize:
            norm = sum([v ** 2 for v in self.v]) ** 0.5
            self.v = tuple((v / norm for v in self.v))
        else:
            # assert norm equals 1.
            assert 1 -EPS < sum((v**2 for v in self.v)) < 1 + EPS
        assert type(self.construction) == list
        for c in self.construction:
            assert self.gateset.is_valid_construction(c), f"bad construction {c} in {self.construction}"

    def _needs_fixup_direction(self):
        """to omit double-counting in SU(2), every variable is inverted if the first non-zero value is negative"""
        for v in self.v:
            if -EPS < v < EPS:
                continue
            if v < 0:
                return True
            else:
                return False
        return False

    @staticmethod
    def from_matrix(matrix, hierarchy=0, construction=[], normalize=True, gateset=NULL_GATESET):
        assert len(matrix) == 2 and len(matrix[0]) == 2
        i = (matrix[0][0] + matrix[1][1]) / 2
        x = (matrix[0][1] + matrix[1][0]) / 2j
        y = (matrix[0][1] - matrix[1][0]) / 2
        z = (matrix[0][0] - matrix[1][1]) / 2j
        # fix phase
        i, x, y, z = Uop._fix_phase((i, x, y, z))

        return Uop(i, x, y, z, hierarchy, construction, True, gateset=gateset)

    @staticmethod
    def _fix_phase(values):
        c = 1.0
        # find the first non-real value and get the conjugate
        for v in values:
            if - EPS < abs(v) < EPS:
                continue
            if v.imag < -EPS or EPS < v.imag:
                c = v.conjugate()
                break
        return [(v*c).real for v in values]

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
        return [[i+z*1j, y+x*1j], [-y+x*1j, i-z*1j]]

    def __matmul__(self, other):
        # see Nielsen & Chuang, Exercise 4.15 (1)
        i1, x1, y1, z1 = self.v
        i2, x2, y2, z2 = other.v
        ni = i1*i2 - x1*x2 - y1*y2 - z1*z2
        nx = i1*x2 + x1*i2 - y1*z2 + z1*y2
        ny = i1*y2 + x1*z2 + y1*i2 - z1*x2
        nz = i1*z2 - x1*y2 + y1*x2 + z1*i2
        return Uop(ni, nx, ny, nz, self.hierarchy + other.hierarchy, self.construction + other.construction, gateset=self.gateset)

    def dagger(self):
        ncon = []
        for c in reversed(self.construction):
            ncon += self.gateset.get_dagger(c)
        return Uop(self.i, -self.x, -self.y, -self.z, self.hierarchy, ncon, gateset=self.gateset)

    def __str__(self):
        nn = math.sqrt(sum((value ** 2 for value in self.v[1:])))
        return ("****\n"
                "#T gate : {self.hierarchy}\n"
                "{self.v[0]:f} I + {self.v[1]:f} iX + {self.v[2]:f} iY + {self.v[3]:f} iZ\n"
                "rot = {rot:f}π, axis = {axis}\n"
                "****".format(self=self,
                              rot= 2 * math.acos(self.v[0])/math.pi,
                              axis=tuple((v/nn if nn > 0 else 0 for v in self.v[1:]))))

    def __hash__(self):
        return hash((self.v, self.hierarchy, tuple(self.construction)))
        
    def construction_str(self):
        return "".join(self.construction)

    def operator_distance(self, other):
        def max_eigen_value(mat):
            p = abs(mat[0][0])**2 + abs(mat[1][0])**2
            q = abs(mat[0][1])**2 + abs(mat[1][1])**2
            r2 = abs(mat[0][0] * mat[0][1].conjugate() + mat[1][0].conjugate() * mat[1][1]) ** 2
            return (p + q + ((p - q)** 2 + 4 * r2) ** 0.5)/2
        u = self.matrix_form()
        v = other.matrix_form()
        for i in range(2):
            for j in range(2):
                u[i][j] -= v[i][j]
        values = [max_eigen_value(u)]
        assert values[-1] >= 0, str(self)
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
    
    def __repr__(self):
        return  "Uop({self.i}, {self.x}, {self.y}, {self.z}, {self.hierarchy}, {self.construction})".format(self=self)
    
    def __eq__(self, other):
        return (self.v == self.v
         and self.construction == self.construction
         and self.hierarchy == self.hierarchy)

    def is_similar(self, other):
        for left, right in zip(self.v, other.v):
            if not (left - EPS < right < left + EPS):
                return False
        return True


def gc_decompose(udd):
    # udd = Rn(θ)
    s = ((1 - udd.i) / 2) ** 0.25
    c = (1 - s ** 2) ** 0.5         # cos Φ  = 1 - sin Φ
    v = Uop(c, s, 0, 0, 0, [], False, gateset=udd.gateset)   # v = cosΦ I - i sinΦ X = Rx(Φ)
    w = Uop(c, 0, s, 0, 0, [], False, gateset=udd.gateset)   # v = cosΦ I - i sinΦ Y = Ry(Φ)

    # rotation axis n = (nx, ny, nz)
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

    s = Uop(0, x, y, z, 0, [], False, gateset=udd.gateset)
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


def execute_solovay_kitaev(u, rec):
    uop_set = u.gateset.epsilon_net()
    return solovay_kitaev(uop_set, u, rec)