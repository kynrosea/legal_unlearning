# ensures we use the same seed when "randomly" selecting or generating anything

# sources:
# https://www.w3schools.com/python/ref_random_seed.asp
# https://numpy.org/doc/2.3/reference/random/generated/numpy.random.seed.html
# https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4

from config import *
import random
import numpy as np
import torch

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

