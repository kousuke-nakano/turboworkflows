#!python -u
# -*- coding: utf-8 -*-

from __future__ import print_function

# python modules
import os, sys
import subprocess

# set logger
from logging import config, getLogger, StreamHandler, Formatter
logger = getLogger('turbo-workflows').getChild(__name__)

# pyturbo module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# turbo-genius related path lists
turbo_workflows_root=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
turbo_workflows_source_root=os.path.join(turbo_workflows_root, 'turboworkflows')
turbo_workflows_tmp_dir=os.path.join(os.path.abspath(os.environ['HOME']), '.turbo_workflows_tmp')

# generate workflows temp. dir.
os.makedirs(turbo_workflows_tmp_dir, exist_ok=True)