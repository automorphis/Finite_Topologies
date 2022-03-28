import itertools
from itertools import product
from math import comb
from pathlib import Path

import numpy as np


def get_inventory_bound(inv, n, m):
    """Calculate an upper bound of `inv[m]`.

    :param inv: type `tuple`.
    :param n: type `int`. The size of the finite topology for the inventory `inv`.
    :param m: type `int`.
    :return: type `int`.
    """
    # Currently we only have the trivial upper bound.
    return n - sum(inv[1:m])

def calc_inventory(top):
     inv = np.zeros(len(top)+1)
     inv[0] = 1
     sizes, counts = np.unique(np.sum(top, axis=2), return_counts=True)
     inv[sizes] = counts
     return inv

def check_homeomorphism(top1, top2, homeo):
    """Short-circuited check for homeomorphism. It is preferred to call this rather than calling
    `top2 == apply_homeomorphism(top1, homeo)`.

    :param top1: type `numpy.ndarray`. Must be square 2-d array.
    :param top2: ditto.
    :param homeo: type `tuple`. Each entry must be a non-negative `int`.
    :return: `True` if `homeo` is indeed a homeomorphism from `top1` to `top2` and `False` otherwise.
    """

    if len(top1) != len(top2) or len(top1) != len(homeo) or len(top1) != len(set(homeo)):
        return False

    else:
        return all(top1[homeo,i] == top2[:,homeo[i]] for i in range(len(top1)))


def apply_homeomorphism(top, homeo):
    """

    :param top: type `numpy.ndarray`. Must be a square 2-d array.
    :param homeo: type `tuple`. Each entry must be a non-negative `int`.
    :return: type `numpy.ndarray`. Square 2-d array.
    """
    pass

def are_homeomorphic(top1, top2, skip_calc_inventory = False):
    """Return `True` if the topologies induced by the minimal bases represented by `top1` and `top2`
    are homeomorphic and `False` otherwise.

    :param top1: type `numpy.ndarray` 0-1 valued array
    :param top2: ditto
    :param skip_calc_inventory: type `bool`, default `False`. If `True`, this function will not call
    `calc_inventory`.
    :return: type `bool`
    """

    if len(top1) != len(top2):
        return False

    elif not skip_calc_inventory and calc_inventory(top1) != calc_inventory(top2):
        return False

    n = len(top1)

    sizes1 = np.sum(top1,axis=2)
    sizes2 = np.sum(top2,axis=2)

    possible_images = [np.where(sizes2 == sizes1[i])[0] for i in range(n)]

    for homeo in itertools.product(*possible_images):
        if check_homeomorphism(top1, top2, homeo) == top2:
            return True

    else:
        return False


class Inventory_Iterator:

    def __init__(self, n):
        self.n = n
        self._curr_inv = (1,) + (0,) * self.n
        self._bounds = [1] + [get_inventory_bound(self._curr_inv, self.n, m) for m in range(1,self.n+1)]
        self._curr_incr_index = 5
        self._raise_stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):

        if self._raise_stop_iteration:
            raise StopIteration

        i = self._curr_incr_index

        finding_increment = True
        while finding_increment:

            if i == 0:
                self._raise_stop_iteration = True
                return self._curr_inv

            elif self._bounds[i] is None:
                first_None_index = self._bounds.index(None)
                for j in range(first_None_index, self.n+1):
                    self._bounds[j] = get_inventory_bound(self._curr_inv, self.n, j)

            if self._curr_inv[i] < self._bounds[i]:
                next_inv = self._curr_inv[:i] + (self._curr_inv[i] + 1,) + (0,) * (self.n - i)
                finding_increment = False

            else:
                self._bounds[i] = None
                i -= 1

        self._curr_incr_index = self.n
        ret = self._curr_inv
        self._curr_inv = next_inv
        return ret

def calc_T0_topologies(n):
    """Calculate all T0 topologies on a fixed finite set of size `n`.

    :param n: type `int`.
    :return: type `numpy.ndarray` of shape `(n,n,K)`, where `K` is the total number of T0 finite topologies.
    """

    possible_open_sets = [np.zeros((n,comb(n,k)), dtype=int) for k in range(n+1)]
    for k, arr in enumerate(possible_open_sets[1:]):
        k+=1
        for i, ones in enumerate(itertools.combinations(range(n),k)):
            arr[ones,i] = 1

    binom_coeffs = [arr.shape[1] for arr in possible_open_sets]

    for inv in Inventory_Iterator(n):
        for m in range(1, n+1):
            for i in itertools.combinations(range(binom_coeffs[m]), inv[m]):pass



calc_T0_topologies(5)


