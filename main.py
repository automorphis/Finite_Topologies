import itertools
from collections import Counter
from itertools import product
from math import comb, log
from pathlib import Path

import numpy as np

def is_subset(a, b, a_len = None):
    if a_len is None:
        a_len = np.sum(a)
    return np.sum(a * b) == a_len

def is_strict_subset(a, b, a_len = None, b_len = None):
    if a_len is None:
        a_len = np.sum(a)
    if b_len is None:
        b_len = np.sum(b)
    intersection_len = np.sum(a * b)
    return intersection_len == a_len and a_len < b_len

def check_represents_T0_topology(top):
    n = len(top)
    topT = np.copy(top).transpose()

    # check dtype
    if top.dtype is not np.dtype('int'):
        return False

    # check is 0-1
    if len(np.where(top == 1)[0]) + len(np.where(top == 0)[0]) != n ** 2:
        return False

    # check contains diagonal
    if np.sum(np.diagonal(top)) != n:
        return False

    # check T0
    for i in range(n):
        open_set = topT[:,i]
        if np.any(np.all(topT[:i, :] == open_set, axis=1)):
            return False

    # check minimal basis property
    for i1, i2 in itertools.combinations(range(n), 2):
        if top[i1,i2] == 1 and not is_subset(top[:,i1], top[:,i2]):
            return False

    return True

def _bitflip_ndarray_inplace(a):
    a *= -1
    a += 1

def convert_to_bit_list(num, length=None):
    if num > 0:
        num_bits = int(log(num,2)+1)
    else:
        num_bits = 1
    if length is not None:
        num_bits = max(num_bits, length)
    return [1 if num & (1 << (num_bits - 1 - n)) else 0 for n in range(num_bits-1,-1,-1)]

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
     sizes, counts = np.unique(np.sum(top, axis=0), return_counts=True)
     inv[sizes] = counts
     return inv

def calc_row_inventory(top):
    inv = np.zeros(len(top) + 1)
    inv[0] = 1
    sizes, counts = np.unique(np.sum(top, axis=1), return_counts=True)
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

    if len(top1) != len(top2) or len(top1) != len(homeo) or len(set(homeo)) != len(top1):
        return False

    else:
        top1 = top1[:,homeo]
        top1 = top1[homeo,:]
        return np.all(top1 == top2)

        # n = len(top1)
        # homeo_mat = np.zeros((n,n),dtype=int)
        # homeo_mat[np.arange(n), homeo] = 1
        # homeo_matT = np.copy(homeo_mat).transpose()
        # return (
        #     np.all(np.matmul(np.matmul(homeo_mat, top1), homeo_matT) == top2) or
        #     np.all(np.matmul(np.matmul(homeo_mat, top2), homeo_matT) == top1)
        # )
        # return (
        #     all(np.all(top1[homeo, i] == top2[:, homeo[i]]) for i in range(len(top1))) or
        #     all(np.all(top2[homeo, i] == top1[:, homeo[i]]) for i in range(len(top1)))
        # )

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

    elif not skip_calc_inventory and np.any(calc_inventory(top1) != calc_inventory(top2)):
        return False

    elif np.any(calc_row_inventory(top1) != calc_row_inventory(top2)):
        return False

    n = len(top1)

    sizes1 = np.sum(top1,axis=0)
    sizes2 = np.sum(top2,axis=0)

    possible_images = [np.where(sizes2 == sizes1[i])[0] for i in range(n)]

    for homeo in itertools.product(*possible_images):
    # for homeo in itertools.permutations(range(n)):
        if check_homeomorphism(top1, top2, homeo):
            return True

    else:
        return False

class Inventory_Iterator:

    def __init__(self, n):
        self.n = n
        self._curr_inv = (1,) + (0,) * (self.n-1) + (self.n,)
        self._bounds = [1] + [get_inventory_bound(self._curr_inv, self.n, m) for m in range(1,self.n+1)]
        self._raise_stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):

        if self._raise_stop_iteration:
            raise StopIteration

        i = self.n

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

        ret = self._curr_inv
        self._curr_inv = next_inv[:-1] + (self.n - sum(next_inv) + 1,)
        return ret

class _Minimal_Open_Set_Iterator:

    def __init__(self, t0_top_iter, i):

        self.i = i
        self.n = t0_top_iter.n
        self.curr_top = t0_top_iter.curr_top
        self.flattened_inv = t0_top_iter.flattened_inv
        self.power_set = t0_top_iter.power_set

        self._minimal_open_set_mask = np.zeros(self.n, dtype=int)
        # intersection of all U_x such that `self.i` is in U_x
        np.all(
            self.curr_top[:, np.nonzero(self.curr_top[self.i,:self.i])[0]],
            axis = 1,
            out = self._minimal_open_set_mask
        )
        self._mask_length = np.sum(self._minimal_open_set_mask)
        self._checked_small_mask = False

        # x such that U_x does not contains `self.i`
        self._far_indices = np.where(self.curr_top[self.i, :self.i] == 0)[0]

        # x such that U_x \subset `self._minimal_open_set_mask` and U_x does not contain `self.i`
        self._carryover_indices = []
        for i in self._far_indices:
            if is_subset(self.curr_top[:,i], self._minimal_open_set_mask, self.flattened_inv[i]):
                self._carryover_indices.append(i)
        self._carryover_indices = np.array(self._carryover_indices, dtype=int)

        self._carryover_mask = np.zeros(self.n, dtype=int)
        self._carryover_mask[self._carryover_indices] = 1

        # if `self._minimal_open_set_mask` matches any previous column of `self._curr_top`
        if self.i > 0:
            self._mask_match = np.any(
                np.all(self.curr_top.transpose()[:self.i, :] == self._minimal_open_set_mask, axis = 1)
            )
        else:
            self._mask_match = False

        self._noncarryover_mask = self._minimal_open_set_mask * (1 - self._carryover_mask)
        self._noncarryover_mask[self.i] = 0

        self._carryover_index = 0
        self._max_carryover_index = 2 ** len(self._carryover_indices) - 1

        self._noncarryover_iter = None
        self.carryover_union = None
        self.num_ones_in_union = None

        # self._cum_union_sizes = np.cumsum(np.sum( 1 - t0_top_iter.leftovers[:,:i], axis = 0))

    def __iter__(self):
        return self

    def __next__(self):

        if (
            not self._checked_small_mask and (
                (self._mask_match and self.flattened_inv[self.i] >= self._mask_length) or
                (not self._mask_match and self.flattened_inv[self.i] > self._mask_length)
            )
        ):
            raise StopIteration

        else:
            self._checked_small_mask = True

        finding_next = True
        while finding_next:

            if self._carryover_index > self._max_carryover_index:
                raise StopIteration

            if self._noncarryover_iter is None:

                subset = self.power_set[:,self._carryover_index]
                size = subset[0]
                indices = subset[1:size+1]

                carryover_cols = self.curr_top[:, self._carryover_indices[indices]]
                self.carryover_union = np.zeros(self.n, dtype=int)
                np.any(carryover_cols, axis = 1, out = self.carryover_union)

                carryover_length = np.sum(self.carryover_union)
                _is_subset = is_subset(
                    self.carryover_union, self._minimal_open_set_mask, carryover_length
                )
                _is_strict_subset = is_strict_subset(
                    self.carryover_union, self._minimal_open_set_mask, carryover_length, self._mask_length
                )

                noncarryover_indices = np.nonzero(self._noncarryover_mask * (1 - self.carryover_union))[0]

                if (
                    (_is_strict_subset or (not self._mask_match and _is_subset)) and
                    carryover_length + len(noncarryover_indices) + 1 >= self.flattened_inv[self.i] > carryover_length
                ):

                    self._noncarryover_iter = itertools.combinations(
                        noncarryover_indices,
                        self.flattened_inv[self.i] - carryover_length - 1
                    )

            if self._noncarryover_iter is not None:
                try:
                    curr_indices = next(self._noncarryover_iter)
                    finding_next = False

                except StopIteration:
                    self._carryover_index += 1
                    self._noncarryover_iter = None

            else:
                self._carryover_index += 1

        curr_indices = np.array(curr_indices, dtype=int)
        non_carryover_portion = np.zeros(self.n, dtype=int)
        non_carryover_portion[curr_indices] = 1
        ret = self.carryover_union + non_carryover_portion
        ret[self.i] = 1
        return ret

class T0_Topology_Iterator:

    def __init__(self, inv, power_set):

        self.inv = inv
        self.n = len(inv)-1

        self.flattened_inv = np.zeros(self.n, dtype=int)
        start_index = 0
        for i in range(1, self.n+1):
            self.flattened_inv[start_index : start_index + self.inv[i]] = i
            start_index += self.inv[i]

        self.curr_top = np.zeros((self.n,self.n), dtype=int)
        # self.leftovers = np.ones((self.n,self.n), dtype=int)

        self.power_set = power_set

        self._iters = [_Minimal_Open_Set_Iterator(self,0)] + [None]*(self.n-1)

        self.next_iter_index = 0

        #
        # self._num_ones = np.sum(self._power_set, axis=2)
        # self._indices = [np.where(self._num_ones == k)[0] for k in range(self.n + 1)]
        #
        # self._one_locs = np.zeros((self.n, 2 ** self.n), dtype=int)
        # for i in range(2 ** self.n):
        #     self._one_locs[:self._num_ones[i], i] = np.nonzero(self._power_set[:, i])[0]

    def __iter__(self):
        return self

    def __next__(self):

        self.curr_top = np.copy(self.curr_top)

        finding_next = True
        while finding_next:

            if self.next_iter_index <= -1:
                raise StopIteration


            for i in range(self.next_iter_index, self.n):

                try:
                    coords = next(self._iters[i])

                except StopIteration:
                    self._iters[self.next_iter_index] = None
                    self.next_iter_index -= 1
                    break

                self.curr_top[:, i] = coords
                # self.leftovers[:, i:] = 1-coords

                if i < self.n-1:
                    self._iters[i+1] = _Minimal_Open_Set_Iterator(self, i+1)
                    self.next_iter_index += 1

            else:
                return self.curr_top


            # for i in range(next_iter_index, self.n):
            #
            #
            #
            # if exist_None_index:
            #
            #     for j in range(first_None_index, self.n):
            #         self._minimal_open_set_mask[:, j] = np.all(
            #             self._minimal_open_set_mask[:, top[j, :]],
            #             axis=0
            #         )
            #
            # else:
            #
            #
            #
            #
            # for j in range(first_None_index, self.n):
            #
            #     self._minimal_open_set_mask[:,j] = np.all(
            #         self._minimal_open_set_mask[:, top[j,:]],
            #         axis=0
            #     )
            #
            #     if np.sum(self._minimal_open_set_mask[:,j]) > self.inv[j]:
            #
            #         self._iters[j] = itertools.combinations(
            #             np.nonzero(self._minimal_open_set_mask[:,j])[0],
            #             self.inv[j]
            #         )
            #
            #     else:
            #         break
            #
            # else:
            #     finding_next = False
            #
            # for i in range(self.n-1,-1,-1):
            #
            #     try:
            #         coords = next(self._iters[i])
            #         top[:,:i] = self._minimal_open_set_mask[:,:i]
            #         top[coords,i] = 1
            #         break
            #
            #     except StopIteration:
            #         self._iters[i] = None
            #
            # else:
            #
            #     if self.curr_top is not None:
            #         self._raise_stop_iteration = True
            #         return self.curr_top
            #     else:
            #         raise RuntimeError

def calc_T0_topologies(n):
    """Calculate all T0 topologies on a fixed finite set of size `n`.

    :param n: type `int`.
    :return: type `numpy.ndarray` of shape `(n,n,K)`, where `K` is the total number of T0 finite topologies.
    """

    power_set = np.zeros((n+1, 2 ** n), dtype=int)
    for i in range(2 ** n):
        bitlist = convert_to_bit_list(i, n)
        size = np.sum(bitlist)
        power_set[0,i] = size
        power_set[1:size+1,i] = np.nonzero(bitlist)[0]


    all_homeo_classes = []

    homeo_classes_by_inv = {}

    for inv in Inventory_Iterator(n):
        homeo_classes_this_inv = []

        for top in T0_Topology_Iterator(inv, power_set):

            if not check_represents_T0_topology(top):
                raise ValueError

            for other_top in homeo_classes_this_inv:
                if are_homeomorphic(top, other_top, True):
                    break

            else:
                homeo_classes_this_inv.append(top)
        # if len(homeo_classes_this_inv) > 0:
        #     for cls in homeo_classes_this_inv:
        #         try:
        #             row_inv = [1] + [0]*n
        #             for size, count in Counter(np.sum(cls, axis=1)).items():
        #                 row_inv[size] = count
        #         except TypeError:
        #             raise TypeError
        #         row_inv = tuple(row_inv)
        #         if (inv, row_inv) in homeo_classes_by_inv:
        #             homeo_classes_by_inv[(inv, row_inv)].append(cls)
        #         else:
        #             homeo_classes_by_inv[(inv, row_inv)] = [cls]
        all_homeo_classes.extend(homeo_classes_this_inv)



    return np.stack(all_homeo_classes, axis=2)

print(calc_T0_topologies(8).shape[2])
# print([(n, calc_T0_topologies(n).shape[2]) for n in range(1, 8)])


