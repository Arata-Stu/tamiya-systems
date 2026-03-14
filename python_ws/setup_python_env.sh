#!/bin/bash
set -e

cd /python_ws/
pip3 install -r requirements.txt
pip3 install -e f1tenth_gym_jax

pip3 uninstall -y jax jaxlib
pip3 install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/dist-packages:/home/admin/.local/lib/python3.10/site-packages
python3 -c "import jax; print('Devices:', jax.devices())"