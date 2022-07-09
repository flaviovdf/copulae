#  -*- coding: utf-8


from distutils.core import setup


import sys


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 10):
        print('Requires Python version 3.10 or later')
        sys.exit(-1)

    with open('requirements.txt') as req_file:
        requires = [line.strip() for line in req_file]

    setup(
        name='musica_dcc_ufmg',
        packages=['copulae'],
        install_requires=requires,
        version='0.01',
        description='Neural Copula Code',
        author='Flavio Figueiredo'
    )
