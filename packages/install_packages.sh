# run this script in drqn/packages
conda create --name pomdpr python=3.7
conda activate pomdpr
pip install numpy scipy torch  gym wandb
cd rl_parsers || exit
pip install -r requirements.txt
pip install .
cd ../gym-pomdps || exit
pip install -r requirements.txt
pip install .