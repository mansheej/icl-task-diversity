# Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

Code for [Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression
](https://arxiv.org/abs/2306.15063)

**Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)**

All experiments were run on TPUs using the Google TPU Research Cloud. 
Apply for TPU access at https://sites.research.google/trc/about/. 

After provisioning a TPU VM, create a Python virtual environment using:
```sh
conda create -n icl -y python=3.10
conda activate icl
```
and install dependencies using
```sh
pip install -r requirements.txt
pip install -e .
```

To train a model, modify `icl/configs/example.py` and then run:
```sh
python run.py --config=icl/configs/example.py
```
