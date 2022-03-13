import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from main import *

print('step14'.center(50, '='))
a = Variable(np.array(2))
y = add(add(a, a), a)
y.backward()
print(f"a.grad: {a.grad}")

a.cleargrad()
print(f"a.grad: {a.grad}")

y = add(add(a, a), add(add(a, a), a))
y.backward()
print(f"a.grad: {a.grad}")
