"""
Unit tests for main module.
"""

import pytest
from src.main import load_config

def test_load_config():
    """Test configuration loading."""
    config = load_config()
    assert 'project' in config
    assert 'data' in config
    assert 'model' in config

def test_config_structure():
    """Test configuration structure."""
    config = load_config()
    assert 'name' in config['project']
    assert 'raw_data_path' in config['data']
    assert 'model_path' in config['model']
