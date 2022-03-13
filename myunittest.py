from main import *
import unittest
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def gradient_check(self):
        x = Variable(np.random.random(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        # print(flg)
        # print(x.grad)
        # print(num_grad) #printが上手くいかないのが不便
        self.assertTrue(flg) #なんか上手くいかない？

a = SquareTest()
a.gradient_check()
unittest.main() #これでも可