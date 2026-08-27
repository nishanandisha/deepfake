import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())

    assert a == b
