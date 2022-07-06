from setuptools import setup

install_requires = [
    'numpy',
    'torch'
]

setup(
    name='moli',
    version='0.1',
    packages=['moli'],
    url='https://github.com/kaboroevich/modrp/moli',
    license='GPLv3',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='MOLI: Multi-Omics Late Integration with deep neural networks'
                ' for drug response prediction',
    install_requires=install_requires,
    extras_require={}
)
