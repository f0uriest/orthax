"""Setup/build/install script for orthax."""

import os

import versioneer
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(os.path.join(here, "requirements-dev.txt"), encoding="utf-8") as f:
    dev_requirements = f.read().splitlines()
dev_requirements = [
    foo for foo in dev_requirements if not (foo.startswith("#") or foo.startswith("-r"))
]

setup(
    name="orthax",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=("Orthogonal polynomials with JAX"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/f0uriest/orthax",
    author="Rory Conlin",
    author_email="wconlin@princeton.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="polynomial orthogonal hermite laguerre legendre chebyshev",
    packages=find_packages(exclude=["docs", "tests", "local", "report"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    project_urls={
        "Issues Tracker": "https://github.com/f0uriest/orthax/issues",
        "Contributing": "https://github.com/f0uriest/orthax/blob/master/CONTRIBUTING.rst",  # noqa: E501
        "Source Code": "https://github.com/f0uriest/orthax/",
        "Documentation": "https://orthax.readthedocs.io/",
    },
)
