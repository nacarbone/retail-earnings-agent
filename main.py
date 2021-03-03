import argparse
import subprocess

import ray

from manager import run_experiment

parser = argparse.ArgumentParser(description='Train and test RL models on the market environment.')
parser.add_argument('exp-type', metavar='e', type=str,
                    help='the type of the experiment to run')
parser.add_argument('chkpt', metavar='c', type=str,
                    help='the path to the checkpoint from which to restore')

 

