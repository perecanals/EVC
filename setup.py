#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="extracranial_vessel_labelling",
      version="1.0",
      description="Extracranial vessel labelling framework: a node classification problem",
      author="Pere Canals",
      author_email="perecanalscanals@gmail.com",
      packages=find_packages(),
      package_data={},
      install_requires=[
        "matplotlib",
        "networkx",
        "numpy",
        "sklearn",
        "torch",
        "torch-geometric"
    ]
)
