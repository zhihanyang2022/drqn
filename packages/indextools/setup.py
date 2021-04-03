#!/usr/bin/env python

import sys

from setuptools import setup

if sys.version_info < (3, 6):
    sys.stderr.write('Python >= 3.6 is required.')
    sys.exit(1)


setup(
    name='indextools',
    version='0.1.0',
    description='indextools - bijective mapping between semantic value and index.',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/bigblindbais/indextools',
    packages=['indextools'],
    package_dir={'': 'src'},
    test_suite='tests',
    license='MIT',
)
