import unittest

import rl_parsers.mdp as dotmdp


class MDP_Test(unittest.TestCase):
    def parse_file(self, fname):
        with open(fname) as f:
            dotmdp.parse(f.read())

    def test_parser(self):
        self.parse_file('tests/mdp/gridworld.mdp')
