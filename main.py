from itertools import product
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
    ranges.append( range(comb(N, k + 1) + 1) )

inventories = []

"""
The idiom `product(*tuple(ranges))` found below is a bit tricky to understand but it is extremely useful.

Recall that the function `product` is a Python standard library function. It takes the Cartesian
product of an arbitrary number of iterables. You can find the docs for `product` here:
https://docs.python.org/3/library/itertools.html#itertools.product

A naive inventory is an element of
`product(ranges[0], ranges[1], ranges[2], ..., ranges[N-1])`

However, we cannot use the `...` syntax in this way in Python. Instead, we must use the unary 
"unpacking" operator, denoted by `*`. This operator takes a tuple and turns each element of the 
tuple into an argument of a function. 

So for example, you could do
`comb(*(5,3))`
which gives you the same number as `comb(5,3)` does. The operator `*` is useless in this context.
However, `*` is very useful when the function can take a variable number of arguments, like
`product` does. 

So `product(*tuple(ranges))` takes the list `ranges`, converts it to a tuple, then "unpacks" it
for the variable-parameter function `product`.  
"""
for inv in product(*tuple(ranges)):
    inventories.append(inv)







