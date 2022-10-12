#!python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_collections_workflows_examples():
    try:
        import collections_workflows_examples
        assert True
    except:
        assert False

def test_convertfort10mol_workflows_examples():
    try:
        import convertfort10mol_workflows_examples
        assert True
    except:
        assert False

def test_lrdmc_ext_workflows_examples():
    try:
        import lrdmc_ext_workflows_examples
        assert True
    except:
        assert False

def test_lrdmc_opt_workflows_examples():
    try:
        import lrdmc_opt_workflows_examples
        assert True
    except:
        assert False

def test_lrdmc_workflows_examples():
    try:
        import lrdmc_workflows_examples
        assert True
    except:
        assert False

def test_prep_workflows_examples():
    try:
        import prep_workflows_examples
        assert True
    except:
        assert False

def test_pyscf_workflows_examples():
    try:
        import pyscf_workflows_examples
        assert True
    except:
        assert False

def test_trexio_workflows_examples():
    try:
        import trexio_workflows_examples
        assert True
    except:
        assert False

def test_vmc_workflows_examples():
    try:
        import vmc_workflows_examples
        assert True
    except:
        assert False

def test_vmcopt_workflows_examples():
    try:
        import vmcopt_workflows_examples
        assert True
    except:
        assert False