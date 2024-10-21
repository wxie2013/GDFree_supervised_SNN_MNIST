import sys, time, os, subprocess, json, importlib, ROOT, shutil,glob, math, socket
import numpy as np
from random import uniform
from pathlib import Path
from distutils.util import strtobool
from ray.experimental import tqdm_ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import brian2
import brian2cuda
from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
from brian2cuda.device import CUDAStandaloneDevice

# model related import
from Brian2.get_brian_dir_path_head import *
input_param = importlib.import_module(path_head+'.jobs_sub.ray_tune.input_param')
general_tools = importlib.import_module(path_head+'.tools.general_tools')
tools_names = [x for x in general_tools.__dict__ if not x.startswith("_")]
globals().update({k: getattr(general_tools, k) for k in tools_names})

get_ipython().run_line_magic('run', os.path.join(brian_dir, 'model/Spike.ipynb'))
get_ipython().run_line_magic('run', os.path.join(brian_dir, 'model/Spike_MNIST_Nlayer.ipynb'))

# for ray tune
os.environ["RAY_memory_monitor_refresh_ms"] = "0" # disable memory monitor
import argparse, ray
import pandas as pd
from ray import train, tune, air
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
import ray.data as rd
import logging

#pyTorch related
import torch, torchmetrics
import numpy as np
from typing import Dict
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.torch import TorchConfig
import torch.distributed as dist
from Brian2.tools.torchvision_data import *

#advserarial attack
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.noise import noise
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
