""" Ensure the tensor key works for all eval functions """
import numpy as np
from jax import numpy as jnp
import pytest
import orthax

EVAL_FUNCS = [
    orthax.chebyshev.chebval,
    orthax.polynomial.polyval,
    orthax.hermite.hermval,
    orthax.hermite_e.hermeval,
    orthax.laguerre.lagval,
    orthax.legendre.legval,
]


@pytest.mark.parametrize("eval_fun", EVAL_FUNCS)
def test_tensor_key(eval_fun):
    """ensure the tensor key in the chebval/polyval/... eval functions
    works with numpy and jax-numpy arrays"""
    a = np.array([[1, 2, 3], [2, 3, 4]])
    b = jnp.array(a)

    assert eval_fun(a, a, tensor=True).shape == (3, 2, 3)
    assert eval_fun(b, b, tensor=True).shape == (3, 2, 3)
    assert eval_fun(a, a, tensor=False).shape == (2, 3)
    assert eval_fun(b, b, tensor=False).shape == (2, 3)
