import brunoflow as bf
from jax import numpy as jnp
import jax
import numpy as np
import torch
import unittest as ut
from . import utils


class LossTestCase(ut.TestCase):
    # needed so that the target/labels are of type int64 --> torch.Long, which is required for many torch loss functions.
    jax.config.update("jax_enable_x64", True)


######################################################

utils.add_tests(
    LossTestCase,
    basename="test_mse_mean",
    fn=lambda x, y: bf.opt.mse_loss(x, y),
    torch_fn=lambda x, y: torch.nn.functional.mse_loss(x, y),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_mse_sum",
    fn=lambda x, y: bf.opt.mse_loss(x, y, reduction="sum"),
    torch_fn=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="sum"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_mse_none",
    fn=lambda x, y: bf.opt.mse_loss(x, y, reduction="none"),
    torch_fn=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="none"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_l1_mean",
    fn=lambda x, y: bf.opt.l1_loss(x, y),
    torch_fn=lambda x, y: torch.nn.functional.l1_loss(x, y),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_l1_sum",
    fn=lambda x, y: bf.opt.l1_loss(x, y, reduction="sum"),
    torch_fn=lambda x, y: torch.nn.functional.l1_loss(x, y, reduction="sum"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_l1_none",
    fn=lambda x, y: bf.opt.l1_loss(x, y, reduction="none"),
    torch_fn=lambda x, y: torch.nn.functional.l1_loss(x, y, reduction="none"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]]),
)

utils.add_tests(
    LossTestCase,
    basename="test_bce_mean",
    fn=lambda x, y: bf.opt.bce_loss(x, y),
    torch_fn=lambda x, y: torch.nn.functional.binary_cross_entropy(x, y),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]], rand=np.random.random),
    test_backward=False,
)

utils.add_tests(
    LossTestCase,
    basename="test_bce_sum",
    fn=lambda x, y: bf.opt.bce_loss(x, y, reduction="sum"),
    torch_fn=lambda x, y: torch.nn.functional.binary_cross_entropy(x, y, reduction="sum"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]], rand=np.random.random),
    test_backward=False,
)

utils.add_tests(
    LossTestCase,
    basename="test_bce_none",
    fn=lambda x, y: bf.opt.bce_loss(x, y, reduction="none"),
    torch_fn=lambda x, y: torch.nn.functional.binary_cross_entropy(x, y, reduction="none"),
    inputs=utils.random_inputs([[(4), (4)], [(2, 3), (2, 3)]], rand=np.random.random),
    test_backward=False,
)

utils.add_tests(
    LossTestCase,
    basename="test_nll_mean",
    fn=lambda x, y: bf.opt.nll_loss(x, y),
    torch_fn=lambda x, y: torch.nn.functional.nll_loss(torch.transpose(x, 1, -1), y),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)

utils.add_tests(
    LossTestCase,
    basename="test_nll_sum",
    fn=lambda x, y: bf.opt.nll_loss(x, y, reduction="sum"),
    torch_fn=lambda x, y: torch.nn.functional.nll_loss(torch.transpose(x, 1, -1), y, reduction="sum"),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)

utils.add_tests(
    LossTestCase,
    basename="test_nll_none",
    fn=lambda x, y: bf.opt.nll_loss(x, y, reduction="none"),
    torch_fn=lambda x, y: torch.nn.functional.nll_loss(torch.transpose(x, 1, -1), y, reduction="none"),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)

utils.add_tests(
    LossTestCase,
    basename="test_crossent_mean",
    fn=lambda x, y: bf.opt.cross_entropy_loss(x, y),
    torch_fn=lambda x, y: torch.nn.functional.cross_entropy(torch.transpose(x, 1, -1), y),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)

utils.add_tests(
    LossTestCase,
    basename="test_crossent_sum",
    fn=lambda x, y: bf.opt.cross_entropy_loss(x, y, reduction="sum"),
    torch_fn=lambda x, y: torch.nn.functional.cross_entropy(torch.transpose(x, 1, -1), y, reduction="sum"),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)

utils.add_tests(
    LossTestCase,
    basename="test_crossent_none",
    fn=lambda x, y: bf.opt.cross_entropy_loss(x, y, reduction="none"),
    torch_fn=lambda x, y: torch.nn.functional.cross_entropy(torch.transpose(x, 1, -1), y, reduction="none"),
    inputs=utils.inputs(
        [
            [jnp.array(np.random.normal(size=(2, 3))), jnp.array(np.full((2), 1))],
            [jnp.array(np.random.normal(size=(2, 2, 3))), jnp.array(np.full((2, 2), 1))],
        ]
    ),
)
