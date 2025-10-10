#!/bin/bash
export PATH=$PATH:/home/runner/.local/bin/

cd /home/runner

# Create Jupyter runtime directory in /tmp (writable location)
export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime
export JUPYTER_DATA_DIR=/tmp/jupyter_data
mkdir -p $JUPYTER_RUNTIME_DIR $JUPYTER_DATA_DIR

echo "entrypoint> Starting..."
# growing-ca train
# # Start Jupyter Notebook server
jupyter notebook --notebook-dir=. --ip=0.0.0.0 --port=8888 --no-browser