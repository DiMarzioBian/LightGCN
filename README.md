# LightGCN-pytorch

This is a Pytorch implementation of SIGIR'20 paper [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126):

Current under construction.

The original PyTorch implementation is [here](https://github.com/gusye1234/LightGCN-PyTorch).


## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.

see more in `dataloader.py`

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

* log output


*NOTE*:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
3. Use `tensorboard` option, it's good.
4. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of *epoch 5* and *epoch 116*).


## Extend:
* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`


## Experiments
