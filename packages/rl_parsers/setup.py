import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.stderr.write('Python >= 3.7 is required.')
    sys.exit(1)

setup(
    name='rl_parsers',
    version='0.1.0',
    description='rl_parsers - parsers for old and new RL file formats',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/rl_parsers',
    packages=find_packages(),
)
