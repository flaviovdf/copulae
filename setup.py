#  -*- coding: utf-8


from setuptools import find_packages
from setuptools import setup


import sys


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 7):
        print('Requires Python version 3.7 or later')
        sys.exit(-1)

    with open('requirements.txt') as req_file:
        requires = [line.strip() for line in req_file]

    setup(
        name='copulae',
        packages=find_packages(),
        install_requires=requires,
        version='0.01',
        description='Neural Copula Code',
        author='Flavio Figueiredo'
    )
