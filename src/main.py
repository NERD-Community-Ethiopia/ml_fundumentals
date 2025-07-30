#!/usr/bin/env python3
"""
Main entry point for the ML project.
"""

import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    """Main function."""
    logger.info("Starting ML project...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Project: {config['project']['name']}")
    
    # Add your main logic here
    logger.info("Project setup complete!")

if __name__ == "__main__":
    main()
