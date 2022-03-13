import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from main import *

print('step12'.center(50, '='))
a = Variable(np.array(2))
b = Variable(np.array(3))
y = add(a, b)
print(y.data)