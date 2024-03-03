
########
orthax
########
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

``orthax`` is a Python package for working with orthogonal (and other) polynomials in JAX.
It largely seeks to replicate the functionality of the ``numpy.polynomial`` package,
through there are some API differences due to limitations of JAX, primarily that
trailing zeros are not automatically trimmed from series, so you should do that
manually if it becomes a concern.

For full details of various options see the `Documentation <https://orthax.readthedocs.io/en/latest/>`__

Installation
============

orthax is installable with ``pip``:

.. code-block:: console

    pip install orthax



.. |License| image:: https://img.shields.io/github/license/f0uriest/orthax?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/f0uriest/orthax/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/709132830.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10035983
    :alt: DOI

.. |Docs| image:: https://img.shields.io/readthedocs/orthax?logo=Read-the-Docs
    :target: https://orthax.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |UnitTests| image:: https://github.com/f0uriest/orthax/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/f0uriest/orthax/actions/workflows/unittest.yml
    :alt: UnitTests

.. |Codecov| image:: https://codecov.io/github/f0uriest/orthax/graph/badge.svg?token=MB11I7WE3I
    :target: https://codecov.io/github/f0uriest/orthax
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/f0uriest/orthax
    :target: https://github.com/f0uriest/orthax/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/orthax
    :target: https://pypi.org/project/orthax/
    :alt: Pypi
