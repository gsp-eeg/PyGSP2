#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='pygsp2',
    version='2.0.3',
    description='Graph Signal Processing in Python',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='EPFL LTS2',
    url='https://github.com/gsp-eeg/pygsp2',
    project_urls={
        'Documentation': 'https://pygsp2.readthedocs.io',
        'Download': 'https://pypi.org/project/pygsp2',
        'Source Code': 'https://github.com/gsp-eeg/pygsp2',
        'Bug Tracker': 'https://github.com/gsp-eeg/pygsp2/issues',
        'Try It Online': 'https://mybinder.org/v2/gh/epfl-lts2/pygsp2/master?urlpath=lab/tree/examples/playground.ipynb',
    },
    packages=[
        'pygsp2',
        'pygsp2.graphs',
        'pygsp2.graphs.nngraphs',
        'pygsp2.filters',
        'pygsp2.tests',
    ],
    package_data={'pygsp2': ['data/pointclouds/*.mat']},
    test_suite='pygsp2.tests.suite',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'networkx',
        'geopy',
        'matplotlib',
        'unidecode',
        'utm',
        'pyxlsb',
        'charset-normalizer',
        'requests',
        'hatchling',
    ],
    extras_require={
        # Optional dependencies for development. Some bring additional
        # functionalities, others are for testing, documentation, or packaging.
        'dev': [
            # Import and export.
            'networkx',
            #'json',
            'utm',
            'geopy',
            'pyxlsb',
            'unidecode',
            # 'graph-tool', cannot be installed by pip
            # Construct patch graphs from images.
            'scikit-image',
            # Approximate nearest neighbors for kNN graphs.
            'pyflann3',
            # Convex optimization on graph.
            'pyunlocbox',
            # Plot graphs, signals, and filters.
            'matplotlib',
            # Interactive graph visualization.
            'pyqtgraph',
            'PyOpenGL',
            'PyQt5',
            # Run the tests.
            'flake8',
            'coverage',
            'coveralls',
            # Build the documentation.
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-gallery',
            'memory_profiler',
            'sphinx-rtd-theme',
            'sphinx-copybutton',
            # Build and upload packages.
            'wheel',
            'twine',
            'requests',
            'hatchling',
            'ruff',
            'codespell',
            'tomli',
            'isort',
            'toml',
            'yapf',
        ],
    },
    license='BSD',
    keywords='graph signal processing',
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
