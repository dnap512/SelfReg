# SelfReg
PyTorch implementation of self-supervised regularization for DG (SelfReg)

## Description
![method](https://user-images.githubusercontent.com/44395361/112263134-26ebff80-8cb2-11eb-9934-d74f44440235.png)
An overview of our proposed SelfReg. Here, we propose to use the self-supervised (in-batch) contrastive losses to regularize the model to learn domain-invariant representations. These losses regularize the model to map the representations of the "same-class" samples close together in the embedding space. We compute the following two dissimilarities in the embedding space: (i) individualized and (ii) heterogenerous self-supervised dissimilarity losses. We further use the stochastic weight average (SWA) technique and the inter-domain curriculum learning (IDCL) to optimize gradients in conflict directions.

## Dependency
- python>=3.6
- pytorch>=1.7.0
- torchvision>=0.8.1


## Easy Run

1. Run `sh download.sh` to download PACS dataset.
2. Open `train.ipynb` and `Run All`.
3. Make sure that the training is running well in the last cell.
4. Check the results stored in path `codes/resnet18/{save_name}/` when the training is completed.
