import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from main import *

print('step13'.center(50, '='))
a = Variable(np.array(2))
b = Variable(np.array(3))
y = exp(square(add(square(a), square(b))))
y.backward()
print(f"y.data: {y.data}")
print(f"y.grad: {y.grad}")
print(f"x.grad: {y.grad}")
