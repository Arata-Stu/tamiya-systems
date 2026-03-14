Setup Python env
```bash
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt

cd ${ISAAC_ROS_WS}/../python_ws/
pip3 install -e f1tenth_gym_jax

pip uninstall -y jax jaxlib && \
pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
```