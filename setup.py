# encoding: utf-8
from setuptools import setup, find_packages

setup(name='stats_method',
      version=__init__.__version__,
      packages=find_packages(exclude=["stock_prediction_ai", "data"]),
      author='miao.lin',
      python_requires='>=3.6',
      platforms='any',
      install_requires=[
          "pandas>=0.24.2",
          "numpy",
          "scikit-learn",
          "xgboost",
          "matplotlib",
          "mxnet",
          "seaborn",
          "statsmodels"
      ]
      )