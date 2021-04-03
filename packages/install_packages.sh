# run these first:
#     conda create --name pomdpr python=3.7
#     conda activate pomdpr
# run this script in drqn/packages
pip install numpy scipy torch gym PyYAML wandb
cd indextools || exit
pip install .
cd ../rl_parsers || exit
pip install -r requirements.txt
pip install .
cd ../gym-pomdps || exit
pip install -r requirements.txt
pip install .