from distutils.core import setup

import setuptools

setup(
    name='pypuf',
    version='0.0.8',
    packages=setuptools.find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nils-wisiol/pypuf",
    license='GNU General Public License Version 3',
    maintainer='Nils Wisiol',
    maintainer_email='pypuf@nils-wisiol.de',
    install_requires=[
        'numpy',
        'tensorflow',
        'scipy',
        'cma-es',
        'pandas',
    ],
)
