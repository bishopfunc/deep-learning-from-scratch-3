import sys, os
sys.path.append(os.pardir)
import numpy as np
import math
from main import *

#step1
print('step1'.center(50, '='))
a = np.array(10)
b = np.array([[1, 2, 3],
                [4, 5, 6]])

scaler = Variable(a)
array = Variable(b)

print(scaler.data)
print(array.data)
assert type(scaler) is Variable
assert type(array) is Variable

# #step2
# print('step2'.center(50, '='))
# a = np.array(10)
# b = np.array([[1, 2, 3], [4, 5, 6]])

# x_a = Variable(a)
# x_b = Variable(b)
# f = Function()
# square_f = Square()
# y_a = square_f(x_a)
# y_b = square_f(x_b)
# print(y_a.data)
# print(y_b.data)
# assert type(y_a) is Variable
# assert type(y_b) is Variable
# assert type(f) is Function
# assert isinstance(square_f, (Variable, Function))
# assert y_a.data == a**2
# assert (y_b.data == b**2).all() #配列比較時

# #step3
# print('step3'.center(50, '='))
# A = Square()
# B = Exp()
# C = Square()
# val = np.array([[1, 2, 3], [4, 5, 6]])
# x = Variable(val)
# a = A(x)
# b = B(a)
# y = C(b)
# print(y.data)
# assert (y.data == (np.exp(val**2))**2).all()

# #step4
# print('step4'.center(50, '='))
# def f(x):
#     A = Square()
#     B = Exp()
#     C = Square()
#     return C(B(A(x)))
# val = np.array([[1, 2, 3], [4, 5, 6]])
# x = Variable(val)
# dy = numerical_diff(f, x)
# print(dy)

# #step5
# print('step5'.center(50, '='))
# print('pass')

# #step6
# print('step6'.center(50, '='))

# # val = np.array([[1, 2, 3], [4, 5, 6]])
# val = np.array(0.5)
# x = Variable(val)

# A = Square()
# B = Exp()
# C = Square()
# def f(x):
#     return C(B(A(x)))
# print(isinstance(x, np.ndarray))
# dy_1 = numerical_diff(f, x)
# print(f'dy_1: {dy_1}')

# def df(x):
#     a = A(x)
#     b = B(a)
#     y = C(b)
#     y.grad = x.grad
#     b.grad = C.backward(y.grad)
#     a.grad = B.backward(b.grad)
#     x.grad = A.backward(a.grad)
#     return x.grad
# x.grad = np.array(1.0)
# dy_2 = df(x)
# print(f'dy_2: {dy_2}')
# assert math.isclose(dy_1, dy_2, rel_tol=1e-5)

# #step7
# print('step7'.center(50, '='))
# A = Square()
# B = Exp()
# C = Square()
# x = Variable(np.array(0.5))
# a = A(x)
# b = B(a)
# y = C(b)
# #7-1
# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# #7-2
# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)
# x1 = x.grad
# #7-3
# x = Variable(np.array(0.5))
# a = A(x)
# b = B(a)
# y = C(b)
# y.grad = np.array(1.0)
# y.backward()
# print(x.grad)
# x2 = x.grad
# assert x1 == x2 #backpropagationの自動化が成功している

# #step8
# print('step8'.center(50, '='))
# print('pass')
# #step7と同じ

# #step9
# print('step9'.center(50, '='))
# #9-1
# x = Variable(np.array(0.5))
# y = square(exp(square(x)))
# y.grad = np.array(1.0)
# y.backward()
# print(x.grad)
# #9-2
# x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
# y = square(exp(square(x)))
# y.backward()
# print(x.grad)
# #9-3
# x = Variable(np.array(1.0))  # OK
# x = Variable(None)  # OK
# try:
#     x = Variable(1.0)  # NG
# except:
#     TypeError
# else:
#     assert False

# #step10
# print('step10'.center(50, '='))
# print('pass')

#step11
print('step11'.center(50, '='))
a = np.array(2)
b = np.array(3)
xs = [Variable(a), Variable(b)]
print(xs[0].data)
add = Add()
ys = add(xs)
y = ys[0]
print(y.data)


