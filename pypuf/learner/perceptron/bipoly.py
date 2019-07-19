from numpy import bincount, array, where
"""
    Implements polynomials over {-1, 1}.
"""

#-----------------------------------------------------------------------
def to_index_notation(mon):
    """
    Converts the internal representation of a monomial (with coefficients and bias)
    to a list of lists containing indices.
    """
    return [list(s) for s in mon if len(s) > 0]

def to_dict_notation(mon):
    res = {frozenset(x):1 for x in mon}
    return res

#-----------------------------------------------------------------------
class BiPoly(object):

    def __init__(self, monomials=None):
        if monomials == None:
            self.monomials = {}
            return
        if type(monomials) == dict:
            self.monomials = monomials
            return
        if type(monomials) == list:
            self.monomials = to_dict_notation(monomials)
            return
        raise Exception("Invalid argument.")

    # -------------------- Container Type Methods --------------------
    def __len__(self):
        return self.monomials.__len__()

    def __length_hint__(self):
        return self.monomials.__length_hint()

    def __getitem__(self, key):
        assert type(key) == frozenset
        return self.monomials.__getitem__(key)

    def __setitem__(self, key, value):
        assert type(key) == frozenset
        assert type(value) == int
        return self.monomials.__setitem__(key, value)

    def __delitem__(self, key):
        assert type(key) == frozenset
        return self.monomials.__delitem__(key)

    def __missing__(self, key):
        assert type(key) == frozenset
        return self.monomials.__missing__(key)

    def __iter__(self):
        return self.monomials.items().__iter__()

    def __contains__(self, key):
        assert type(key) == frozenset
        return self.monomials.__contains__(key)

    def get(self, key):
        assert type(key) == frozenset
        return self.monomials.get(key)

    # -------------------- Numeric Type Methods --------------------
    def __add__(self, other):
        res = self.copy()
        for m, c in other:
            res[m] = (res.get(m) or 0) + c
        return res

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        res = BiPoly()
        for m1, c1 in self:
            for m2, c2 in other:
                m = m1.symmetric_difference(m2)
                c = (res.get(m) or 0) + c1 * c2
                res[m] = c
        return res

    def __neg__(self):
        res = self.copy()
        for m, _ in res:
            res[m] = - res[m]
        return res

    def __str__(self):
        return ' + '.join(
            [
                f'{val:3}' +
                ''.join(['x' + ''.join([chr(0x2080 + int(cc)) for cc in
str(c)]) for c in coeff])
                for coeff, val in self
            ]
        )

    def pow(self, k):
        if k == 1:
            return self.copy()
        # k is not computed yet -> split in two halves
        k_star = k // 2
        A = self.pow(k_star)
        res = A.__mul__(A)
        # If k was uneven, we need to multiply with the base poly again
        if k % 2 == 1:
            res = self.__mul__(res)
        return res

    # -------------------- Custom Methods --------------------
    def copy(self):
        return BiPoly(self.monomials.copy())

    def deg(self):
        return max(self.degrees())

    def degrees(self):
        return array(list(map(len, list(self.monomials.keys()))))

    def degrees_count(self):
        return bincount(self.degrees)

    def low_degrees(self, up_to_degree):
        sorted_keys = array(list(self.monomials.keys()))
        pos = where(self.degrees() < up_to_degree)[0]
        return BiPoly({key:self[key] for key in sorted_keys[pos]})

    def coef_dist(self):
        return bincount(array(list(self.monomials.values())))
#-----------------------------------------------------------------------
