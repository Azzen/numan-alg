import unittest
import na.bisection as bisect
import numpy as np


class TestBisectionMethods(unittest.TestCase):

    def test_simple_bisection_mid_point(self):
        solution, iterations = bisect.find_root(lambda x: np.log(x ** 2) + x - 9, 0.2, 10, 1e-7,
                                                bisect.mid_point_split, full_output=True)
        self.assertEqual(5.566475702822208, solution)
        self.assertEqual(26, iterations)

    def test_interval_bisection_secant(self):
        solutions = bisect.find_roots(lambda x: np.cos(1 / x), -1, -0.05, 0.01, 1e-6, bisect.secant_split)
        self.assertEqual([-0.6366197723675813, -0.2122065907891938, -0.12732395447351627,
                         -0.09094568176679733, -0.07073553026306459, -0.0578745247606892], solutions)
