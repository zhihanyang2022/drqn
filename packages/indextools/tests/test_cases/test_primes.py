import unittest

import indextools


def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, n))


class PrimeTest(unittest.TestCase):
    def test_primes(self):
        int_space = indextools.RangeSpace(50)
        prime_space = indextools.SubSpace(int_space, is_prime)
        values = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)

        self.assertEqual(prime_space.nelems, len(values))
        self.assertCountEqual(prime_space.values, values)
