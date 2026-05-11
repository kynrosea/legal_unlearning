import copy
import json
import random
import string
import csv
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt