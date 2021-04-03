import unittest

import rl_parsers.fsc as dotfsc


class FSC_Test(unittest.TestCase):
    def parse_file(self, fname):
        with open(fname) as f:
            dotfsc.parse(f.read())

    def test_parser(self):
        self.parse_file('tests/fsc/tiger.optimal.fsc')
        self.parse_file('tests/fsc/loadunload.optimal.fsc')
