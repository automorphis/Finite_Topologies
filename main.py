from math import comb

""" `N` is the size of the underlying set of the topology. """
N = 10

"""" `X` is the underlying set of the topology. """
X = list(range(N))

"""
An inventory is a list of non-negative integers of length `N`.

If `inv` is an inventory and `1 <= k <= N`, then `inv[k-1]` is the number of open sets in
a minimal basis of size `k`. 

The Python standard library function `comb` calculates binomial coefficients. Follow this
link for a description:
https://docs.python.org/3/library/math.html#math.comb

Naively, `inv[k-1]` can take on any value between 0 and `comb(N,k)`, inclusive.

The following code creates a list `inventories` of all possible naive inventories.
"""

ranges = []

for k in range(N):
    ranges.append( range(comb(N,k) + 1) )

inventories = []

for rng in ranges:

    inv = []
    for x in rng:
        inv.append(x)

    inventories.append(inv)







