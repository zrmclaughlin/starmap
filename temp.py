from sympy import *
import numpy as np

# a = np.arange(100).reshape(10, 10)
# n1, n2 = np.arange(5), np.arange(5)
# print(a)

# Not what you want
# b = a[n1, n2]  # array([ 0, 11, 22, 33, 44])
# print(b)

# What you want, but only for simple sequences
# Note that no copy of *a* is made!! This is a view.
# b = a[:5, :5]
# print(b)
#
# n1, n2 = np.arange(0, 10), np.arange(7, 10)
# What you want, but probably confusing at first. (Also, makes a copy.)
# np.meshgrid and np.ix_ are basically equivalent to this.
# b = a[n1[:,None], n2[None,:]]
# print(b)

# M = np.eye(6)
# M[1, 4] = 20
# M[3, 2] = 4
# M[2, 1] = 12
# M[5, 5] = 8
#
# sym_M = Matrix(M)
#
# initial_d_dv1, initial_d_dv2, initial_d_dv3 = symbols('initial_d_dv1 initial_d_dv2 initial_d_dv3', real=True)
# final_d_dp1, final_d_dv1, final_d_dp2, final_d_dv2, final_d_dp3, final_d_dv3 = \
#     symbols('final_d_dp1, final_d_dv1 final_d_dp2, final_d_dv2 final_d_dp3, final_d_dv3', real=True)
#
# x = Matrix(6, 1, [0, 0, 0, initial_d_dv1, initial_d_dv2, initial_d_dv3])
#
# print(sym_M*x)


l1, l2, lam = symbols('l1 l2 lam', real=True)
L = Matrix([l1, l2])
C = Matrix([1, 0]).T
print(L * C)
print(L)

E = Matrix([[-l1-lam, 1], [-1, -l2 - lam]])
print(E.det())