#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["h5py", "numpy", "scipy", "arf"]

test_requirements = ["pytest"]

setup(
    name='openephys',
    version='0.1.0',
    description=
    "A set of functions and programs for working with data collected with the Open-Ephys GUI.",
    long_description=readme,
    author="Kyler Brown",
    author_email='kylerjbrown@gmail.com',
    url='https://github.com/kylerbrown/openephys',
    packages=[
        'openephys',
    ],
    package_dir={'openephys': 'openephys'},
    include_package_data=True,
    install_requires=requirements,
    scripts=["openephys/kwik2arf.py", "openephys/kwik2dat.py",
             "openephys/kwik2wav.py"],
    license="MIT",
    zip_safe=False,
    keywords='openephys',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements)
