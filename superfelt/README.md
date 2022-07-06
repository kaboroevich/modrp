# Super.FELT

A python package implementing the method published in [Super.FELT: supervised 
feature extraction learning using triplet loss for drug response prediction 
with multi-omics data][1] by Park, S., Soh, J. & Lee, H. (2021) and based on 
code available at the GitHub repository [DMCB-GIST/Super.FELT][2].

## Classes and Functions

### nn.SuperFeltNet

A custom torch.nn.Module combining the three omic (expression, mutation, cna) 
supervised encoders and the classification layer into a single network. The 
network trains the supervised encoders and classification later sequentially
through defining the number of training epochs for each supervised encoder. 
Once all encoders have stopped training, the classifier is trained for the 
remaining epochs.

```python
class nn.MoliNet(exp_in, exp_out, exp_dr, exp_ep mut_in, mut_out, 
mut_dr, mut_ep, cna_in, cna_out, cna_dr, cna_ep, cls_dr)
```
The output is a tuple where the first three elements are the output of the
supervised encoders (expression, mutation, cna) and the fourth element is 
the output of the classifier.

#### Parameters
* **(exp|mut|cna)_in**: input size to the omic supervised encoder layer
* **(exp|mut|cna)_out**: hidden layer size of the omic supervised encoder layer
* **(exp|mut|cna)_dr**: decay rate of the omic supervised encoder layer
* **(exp|mut|cna)_ep**: number of epochs (calls to forward function) to train 
the omic supervised encoder layer
* **cls_dr**: decay rate of the classifer layer

### optim.SuperFeltAdagrad

A custom torch.optim.Adagrad optimizer for use with Super.FELT.

```python
class optim.MoliAdagrad(net, exp_lr, exp_wd, mut_lr, mut_wd, 
cna_lr, cna_wd, cls_lr, cls_wd)
```

#### Parameters
* **net**: MoliNet network
* **(exp|mut|cna)_lr**: learning rate of the omic supervised encoder
* **(exp|mut|cna)_wd**: weight decay of the omic supervised encoder
* **cls_lr**: learning rate of the classifier
* **cls_wd**: weight decay of the classifier

## Examples

* Super.FELT with reported optimized hyperparameters - 
[superfelt_reported_optima.ipynb](./examples/superfelt_reported_optima.ipynb)

[1]: https://doi.org/10.1186/s12859-021-04146-z
[2]: https://github.com/DMCB-GIST/Super.FELT