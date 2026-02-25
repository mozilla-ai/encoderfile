#!/bin/bash
set -e

# Navigate to the crate directory
cd encoderfile-py
source /opt/venv/bin/activate

# Build the wheels
maturin build --release --out /output
