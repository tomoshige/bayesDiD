# setup.py

from setuptools import setup, find_packages

setup(
    name='bayesDiD',
    version='0.1.0',
    description='A Bayesian difference-in-difference module with effect heterogeneity.',
    author='Tomoshige Nakamura',        # ここはご自身の名前に
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.7'
)
