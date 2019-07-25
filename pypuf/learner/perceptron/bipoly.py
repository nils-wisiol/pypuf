"""
This module provides a data type that represents polynomials over {-1, 1}: BiPoly
It also provides a class that bundles the generation of commonly used BiPoly's: BiPolyFactory
"""
from numpy import bincount, array, where

def to_dict_notation(mon):
    """
    Converts a list of lists containing indices to the internal representation.
    """
    res = {frozenset(x):1 for x in mon}
    return res

class BiPoly(object):
    """
    Implements polynomials over {-1, 1}.
    In this class we use a internal representation of monomials:
    {set(indices) : coefficients}
    This means that {set(1,2,3) : 2, set(0) : 5} is interpreted as:
    2*X1*X2*X3 + 5*X0
    Each set in this dict is considered a monomial.

    Each variable of a monomials is in {-1, 1}, this means that X**2 = 1, hence we can
    eliminate these terms and shorten the monomials.
    """

    def __init__(self, monomials=None):
        """
        :param monomials: list or dict
                          Initialize a BiPoly with list or dict of indices.
        """
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

# -------------------------- Container Type Methods --------------------------
    def __len__(self):
        """
        Returns length of polynomial (number of sums).
        """
        return self.monomials.__len__()

    def __length_hint__(self):
        """
        Returns approximate length of polynomial (for optimizations).
        """
        return self.monomials.__length_hint()

    def __getitem__(self, mon):
        """
        Returns coefficient of a monomial.
        :param mon: frozenset
                    Monomial as a frozenset of indices.
        """
        assert type(mon) == frozenset
        return self.monomials.__getitem__(mon)

    def __setitem__(self, mon, coef):
        """
        Sets coefficient of a monomial.
        :param mon: frozenset
                    Monomial as a frozenset of indices.
        :param coef: int
                      Coefficient of monomial.
        """
        assert type(mon) == frozenset
        assert type(coef) == int
        return self.monomials.__setitem__(mon, coef)

    def __delitem__(self, mon):
        """
        Deletes a monomial-coefficient pair from the BiPoly.
        :param mon: frozenset
                    Monomial to be deleted as a frozenset of indices.
        """
        assert type(mon) == frozenset
        return self.monomials.__delitem__(mon)

    def __iter__(self):
        """
        Return Iterable of this BiPoly.
        Items are pairs of (monomial, coefficient).
        """
        return self.monomials.items().__iter__()

    def __contains__(self, mon):
        """
        Returns whether a monomial is contained in the BiPoly
        :param mon: frozenset
                    Monomial as a frozenset of indices.
        """
        assert type(mon) == frozenset
        return self.monomials.__contains__(mon)

    def get(self, mon):
        """
        Wrapper around dict.get.
        :param mon: frozenset
                    Monomial as a frozenset of indices.
        """
        assert type(mon) == frozenset
        return self.monomials.get(mon)

# -------------------------- Numeric Type Methods --------------------------
    def __add__(self, other):
        """
        Returns the sum of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is added to self.
        """
        res = self.copy()
        for m, c in other:
            res[m] = (res.get(m) or 0) + c
        return res

    def __sub__(self, other):
        """
        Returns the difference of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is substracted from self.
        """
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        """
        Returns the product of this BiPoly and other BiPoly.
        :param other: BiPoly
                      BiPoly that is multiplied to self.
        """
        res = BiPoly()
        for m1, c1 in self:
            for m2, c2 in other:
                m = m1.symmetric_difference(m2)
                c = (res.get(m) or 0) + c1 * c2
                res[m] = c
        return res

    def __neg__(self):
        """
        Returns the negation of this BiPoly (i.e. 0 - self).
        """
        res = self.copy()
        for m, _ in res:
            res[m] = - res[m]
        return res

    def __str__(self):
        """
        Returns a string that represents this polynomial.
        """
        return ' + '.join([
                f'{val:3}' +
                ''.join(['x' + ''.join([chr(0x2080 + int(cc)) for cc in
str(c)]) for c in coeff])
                for coeff, val in self
            ])

    def pow(self, k):
        """
        Returns the exponentiation of this polynomial to the power of integer k.
        :param k: int
                  Power to which this BiPoly will be raised.
        """
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

# -------------------------- Custom Methods --------------------------
    def copy(self):
        """
        Returns a copy of this BiPoly.
        """
        return BiPoly(self.monomials.copy())

    def deg(self):
        """
        Returns the degree of this BiPoly.
        """
        return max(self.degrees())

    def degrees(self):
        """
        Returns an array of all degrees of this BiPoly.
        """
        return array(list(map(len, list(self.monomials.keys()))))

    def degrees_count(self):
        """
        Returns an array of degree-occurances, where value v of index i means
        that the degree i occurs v times.
        """
        return bincount(self.degrees)

    def low_degrees(self, up_to_degree):
        """
        Returns a copy of this BiPoly, that contains only monomials with degree smaller
        than up_to_degree.
        :param up_to_degree: int
                             Only monomials smaller than up_to_degree will be returned.
        """
        sorted_keys = array(list(self.monomials.keys()))
        pos = where(self.degrees() < up_to_degree)[0]
        return BiPoly({key:self[key] for key in sorted_keys[pos]})

    def coef_dist(self):
        """
        Returns an array of coefficient-occurances, where value v of index i means
        that the coefficient i occurs v times.
        """
        return bincount(array(list(self.monomials.values())))

    def to_index_notation(self):
        """
        Returns list of lists, where each list represents a monomial and its entries
        are the indices. Note that coefficients will be omitted.
        """
        return [list(s) for s, _ in self if len(s) > 0]

    def substitute(self, mapping):
        """
        Returns BiPoly that corresponds to substituting each variable by a monomial
        defined in mapping.
        :param mapping: list of lists
                        list of lists where indices in list i with replace Xi in self.
        """
        # For each monomial in self, substitute the entry i with monimial i of mapping
        new_poly = BiPoly()
        for mon, coeff in self:
            new_vars = frozenset()
            for index in mon:
                new_vars = new_vars.symmetric_difference(frozenset(mapping[index]))
            new_poly[new_vars] = (new_poly.get(new_vars) or 0) + coeff
        return new_poly


class BiPolyFactory():
    """
    Collection of functions to build commonly used polynomials over {-1,1}.
    """

    @staticmethod
    def monomials_id(n):
        """
        Returns BiPoly where each variable occurs in its identity, i.e. x1+x2+x3+...
        :param n: int
                  Length of the BiPoly.
        """
        return BiPoly([[i] for i in range(n)])

    @staticmethod
    def monomials_atf(n):
        """
        Returns a BiPoly that corresponds to the internal representation of a
        linearized Arbiter PUF.
        :param n: int
                  Length of the Arbiter PUF.
        """
        return BiPoly(list(range(i,n)) for i in range(n))

    @staticmethod
    def xor_arbiter_monomials(n, k):
        """
        Returns the linearized BiPoly that corresponds to a n-bit k-XOR Arbiter PUF.
        :param n: int
                  Length of the Arbiter PUFs.
        :param k: int
                  Amount of Arbiter PUFs which will be XOR'd.
        """
        mono = MonomialFactory.monomials_atf(n)
        res = mono.pow(k)
        return res

    @staticmethod
    def monomials_ipuf(n, k_up, k_down, m_up=None):
        """
        Returns the linearized BiPoly that corresponds to a n-bit k-XOR Arbiter PUF.
        :param n: int
                  Length of the Arbiter PUFs.
        :param k_up: int
                  Amount of Arbiter PUFs in the upper chain.
        :param k_down: int
                  Amount of Arbiter PUFs in the lower chain.
        :param m_up: int
                  Optional precomputed BiPoly corresponding to the upper chain.
        """
        m_up = m_up or MonomialFactory.xor_arbiter_monomials(n, k_up)
        group_1 = BiPoly()
        for i in range(n//2):
            group_1 = group_1 + (BiPoly(list(range(i, n)) * m_up))

        group_2 = m_up.copy()
        group_3 = BiPoly([list(range(i-1, n))
                            for i in range(n//2+2, n+1)])
        m_down = group_1 + group_2 + group_3
        return m_down.pow(k_down)
