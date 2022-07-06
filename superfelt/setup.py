from setuptools import setup

install_requires = [
    'numpy',
    'torch',
    'moli'
]

setup(
    name='superfelt',
    version='0.1',
    packages=['superfelt'],
    url='https://github.com/kaboroevich/modrp/superfelt',
    license='GPLv3',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='Super.FELT: supervised feature extraction learning using'
                ' triplet loss for drug response prediction with' 
                ' multi-omics data',
    install_requires=install_requires,
    extras_require={}
)
