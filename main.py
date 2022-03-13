from matplotlib.pyplot import cla
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None #親子
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #同じ要素数の中身が1

        funcs = [self.creator]
        #再帰->ループ
        while funcs:
            f = funcs.pop() #末尾から取得
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad) #y.gradで考える
            #再帰 f is Noneまで x.gradはすでに代入済み
            if x.creator is not None:
                funcs.append(x.creator)



class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs] #list
        ys = self.forward(*xs) #self.forward(x0, x1)とおなじ
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] #ysをVariableクラスに適用
        for output in outputs:
            output.set_creator(self) 
        self.inputs = inputs #親子
        self.outputs = outputs #親子
        return outputs if len(outputs) > 1 else outputs[0] #帰り値の長さ

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

        
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))  #ここも変更
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)
