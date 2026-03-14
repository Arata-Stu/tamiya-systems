cd /python_ws/
pip3 install -r requirements.txt
pip3 install -e f1tenth_gym_jax

pip uninstall -y jax jaxlib && \
pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 