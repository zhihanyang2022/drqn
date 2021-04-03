import unittest

import rl_parsers.pomdp as dotpomdp


class POMDP_Test(unittest.TestCase):
    def parse_file(self, fname):
        with open(fname) as f:
            dotpomdp.parse(f.read())

    def test_parser(self):
        self.parse_file('tests/pomdp/tiger.pomdp')
        self.parse_file('tests/pomdp/loadunload.pomdp')
        self.parse_file('tests/pomdp/heaven-hell.pomdp')
