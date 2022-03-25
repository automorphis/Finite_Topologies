from itertools import product
from math import comb

def get_inventory_bound(inv, n, m):
    """Calculate an upper bound of `inv[m]`.

    :param inv: type `tuple`.
    :param n: type `int`. The size of the finite topology for the inventory `inv`.
    :param m: type `int`.
    :return: type `int`.
    """
    # Currently we only have the trivial upper bound.
    return n - sum(inv[1:m])

class inventories:

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





