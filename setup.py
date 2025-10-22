from distutils.core import setup

import setuptools
import platform

tf_dep = (
    "tensorflow-macos>=2.15,<2.21"
    if platform.system() == "Darwin"
    else "tensorflow-cpu>=2.15,<2.21"
)

extras_require = {
    "dev": ["pytest", "xdoctest", "sphinx"],
    "mac-metal": ["tensorflow-metal>=0.1.0"],
}

setup(
    name='pypuf',
    version='3.2.1',
    packages=setuptools.find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nils-wisiol/pypuf",
    license='GNU General Public License Version 3',
    maintainer='Nils Wisiol',
    maintainer_email='pypuf@nils-wisiol.de',
    extras_require=extras_require,
    install_requires=[
        'memory_profiler',
        'numpy>=1.18,<1.26',
        'scipy>=1.5',
        'more_itertools',
        tf_dep,
    ],
)

