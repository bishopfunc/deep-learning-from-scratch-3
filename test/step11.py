
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from main import *

print('step11'.center(50, '='))
a = Variable(np.array(2))
b = Variable(np.array(3))
xs = a, b
print(xs[0].data)
add = Add()
ys = add(xs)
y = ys[0]
print(y.data)