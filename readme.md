# ROCO

This is the official implementation of our ICLR 2023 paper "ROCO: A General Framework for Evaluating Robustness of Combinatorial Optimization Solvers on Graphs".

## Abstract

Solving combinatorial optimization (CO) on graphs has been attracting increasing interests from the machine learning community whereby data-driven approaches were recently devised to go beyond traditional manually-designated algorithms. In this paper, we study the robustness of a combinatorial solver as a blackbox regardless it is classic or learning-based though the latter can often be more interesting to the ML community. Specifically, we develop a practically feasible robustness metric for general CO solvers. A no-worse optimal cost guarantee is developed as such the optimal solutions are not required to achieve for solvers, and we tackle the non-differentiable challenge in input instance disturbance by resorting to black-box adversarial attack methods. Extensive experiments are conducted on 14 unique combinations of solvers and CO problems, and we demonstrate that the performance of state-of-the-art solvers like Gurobi can degenerate by over 20% under the given time limit bound on the hard instances discovered by our robustness metric, raising concerns about the robustness of combinatorial optimization solvers.

## Environment set up
Install required pacakges:
```shell
conda create -n roco python=3.7
pip install tensorflow==1.15
pip install scipy
export TORCH=1.9.0
export CUDA=cu111
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit=11.1
pip install --no-index --upgrade torch_cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --upgrade torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric==1.7.2
pip install tensorboard
pip install networkx==2.2
pip install ortools
pip install texttable
pip install pyyaml
```

## Run Experiments
### DAG Scheduling
**Training:**
```shell
python dag_scheduler_ppo_pytorch.py --scheduler_type sft --num_init_dags 50
```
**Testing:**
```shell
python evaluate_dag.py --scheduler_type sft --num_init_dags 50
```
Our code will automatch the related pretrained model, or you can specify the test_model_weight argument.
The optional solvers can be sft/cp/ts and the optional scales can be 50/100/150.

### ATSP Scheduling
Install LKH-3 which is required by this experiment:
```shell
pip install lkh
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
tar xvfz LKH-3.0.6.tgz
cd LKH-3.0.6
make
```
And you should find an executable at `./LKH-3.0.6/LKH`, which will be called by our code.

**Training**:
```shell
python tsp_ppo_pytorch.py
```
**Testing**:
```shell
python evaluate_tsp.py
```

## Citation and Credits
If you find our paper/code useful in your research, please citing: Coming soon.

