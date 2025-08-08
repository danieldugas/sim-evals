set -e
set -x

conda create -y --prefix=/fsx-cortex/shared/conda_envs/isaac-droid python=3.10
conda activate /fsx-cortex/shared/conda_envs/isaac-droid
conda install -c conda-forge libglu
pip install uv
# install isaacsim 4.5
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# install frozen isaaclab
# modified script to print out the target python path, and use uv
cd ~/Code/robot_world_models/projects
git clone --recurse-submodules git@github.com:danieldugas/sim-evals.git
cd ~/Code/robot_world_models/projects/sim-evals
cp isaaclab.sh ./submodules/IsaacLab/isaaclab.sh
./submodules/IsaacLab/isaaclab.sh -i
# install rest of deps
# converted uv.lock into requirements.txt
uv pip install -r requirements.txt
# Download assets
aws s3 cp s3://openpi-assets-simeval/env_assets/simple_example/assets.zip ./ --no-sign-request
unzip assets.zip

# Run
# python run_eval.py
