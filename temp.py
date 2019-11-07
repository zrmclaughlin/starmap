import numpy as np

import numpy as np

a = np.arange(100).reshape(10, 10)
n1, n2 = np.arange(5), np.arange(5)
print(a)

# Not what you want
b = a[n1, n2]  # array([ 0, 11, 22, 33, 44])
print(b)

# What you want, but only for simple sequences
# Note that no copy of *a* is made!! This is a view.
b = a[:5, :5]
print(b)

n1, n2 = np.arange(5, 10), np.arange(5, 10)
# What you want, but probably confusing at first. (Also, makes a copy.)
# np.meshgrid and np.ix_ are basically equivalent to this.
b = a[n1[:,None], n2[None,:]]
print(b)