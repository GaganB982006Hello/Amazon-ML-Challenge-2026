#!/usr/bin/env python3
"""Root entry point for executing the Amazon ML Challenge 2026 Entity Resolution pipeline."""

import os
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from pipeline import main

if __name__ == "__main__":
    main()
