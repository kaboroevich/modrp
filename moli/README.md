# MOLI

A python package implementing the MOLI method proposed in [MOLI: multi-omics late 
integration with deep neural networks for drug response prediction][1] by 
Sharifi-Noghabi *et al.* (2019).



## Classes and Functions

### nn.MoliNet

A custom torch.nn.Module combining the three omic (expression, mutation, cna) 
supervised encoders and the classification layers into a single network. 

```python
class nn.MoliNet(exp_in, exp_out, exp_dr, mut_in, mut_out, mut_dr,
cna_in, cna_out, cna_dr, cls_dr)
```

The output is a tuple where the first element is the concatenated output of the
supervised encoders and the second element is the output of the classifier.

#### Parameters
* **(exp|mut|cna)_in**: input size to the omic supervised encoder layer
* **(exp|mut|cna)_out**: hidden layer size of the omic supervised encoder layer
* **(exp|mut|cna)_dr**: decay rate of the omic supervised encoder layer
* **cls_dr**: decay rate of the classifer layer

### optim.MoliAdagrad

A custom torch.optim.Adagrad optimizer for use with MoliNet.

```python
class optim.MoliAdagrad(net, exp_lr, mut_lr, cna_lr, cls_lr, cls_wd)
```

#### Parameters
* **net**: MoliNet network
* **(exp|mut|cna)_lr**: learning rate of the omic supervised encoder
* **cls_lr**: learning rate of the classifier
* **cls_wd**: weight decay of the classifier

### loss.moli_combination_loss

```python
def loss.moli_combination_loss(input, target, embedding, margin, gamma) 
```

A combined cost function consisting of a triplet loss and a binary cross-entropy loss.

Loss function used in MOLI. Combines BCE_loss and triplet loss. Triplets are selected
using the method described in [adambielski/siamese-triplet][2]. Note that triplet 
selection is not complete as stated as it does not select anchor-positive pairs 
bidirectionally.

### utils.moli_dataloader

```python
def utils.moli_dataloader(dataset, batch_size)
```

Generates a torch batched DataLoader with weighted sampling as described in the MOLI
paper. Dataset should be a torch TensorDataset built from (expression, mutation,
cna, y) tensors.

## Examples

* MoliNet with reported optimized hyperparameters - 
[reported_optima.ipynb](./examples/reported_optima.ipynb)


[1]: https://doi.org/10.1093/bioinformatics/btz318
[2]: https://github.com/adambielski/siamese-triplet