from numpy import bincount, array, where
"""
    Implements polynomials over {-1, 1}.
    In this class we use a internal representation of monomials:
    {set(indices) : coefficients}
    This means that {set(1,2,3) : 2, set(0) : 5} is interpreted as:
    2*X1*X2*X3 + 5*X0

    Each variable X is in {-1, 1}, this means that X**2 = 1, hence we can eliminate
    these terms and shorten the monomials.
"""

#-----------------------------------------------------------------------

def to_dict_notation(mon):
    """
    Converts a list of lists containing indices to the internal representation.
    """
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

    def to_index_notation(self):
        return [list(s) for s, _ in self if len(s) > 0]

    def substitute(self, mapping):
        """
        Returns BiPoly that corresponds to substituting each variable by a monomial.
        params : mapping needs to be a ordered list with Xi at index i.
        """
        # For each monomial in self, substitute the entry i with monimial i of mapping
        new_poly = BiPoly
        for mon, coeff in self:
            new_vars = frozenset()
            for index in mon:
                new_vars = new_vars.symmetric_difference(frozenset(mapping[index]))
            new_poly[new_vars] = (new_poly.get(new_vars) or 0) + coeff
        return new_poly

    #-----------------------------------------------------------------------
