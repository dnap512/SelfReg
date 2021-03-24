# SelfReg
PyTorch implementation of Self-supervised Contrastive Regularization for Domain Generalization (SelfReg)

## Description
![method](https://user-images.githubusercontent.com/44395361/112263134-26ebff80-8cb2-11eb-9934-d74f44440235.png)
An overview of our proposed SelfReg. Here, we propose to use the self-supervised (in-batch) contrastive losses to regularize the model to learn domain-invariant representations. These losses regularize the model to map the representations of the "same-class" samples close together in the embedding space. We compute the following two dissimilarities in the embedding space: (i) individualized and (ii) heterogenerous self-supervised dissimilarity losses. We further use the stochastic weight average (SWA) technique and the inter-domain curriculum learning (IDCL) to optimize gradients in conflict directions.

![tsne](https://user-images.githubusercontent.com/44395361/112265547-01f98b80-8cb6-11eb-84d8-e62de4eda247.png)
Visualizations by t-SNE for (a) baseline (no DG techniques), (b) [RSC](https://arxiv.org/abs/2007.02454), and (c) ours. For better understanding, we also provide sample images of house from all target domains. Note that we differently color-coded each points according to its class. (Data: [PACS](https://domaingeneralization.github.io/#data))

## Dependency
- python >= 3.6
- pytorch >= 1.7.0
- torchvision >= 0.8.1
- gdown

## How to Use

1. `cd codes/` and `sh download.sh` to download PACS dataset.
2. Open `train.ipynb` and `Run All`.
3. Make sure that the training is running well in the last cell.
4. Check the results stored in path `codes/resnet18/{save_name}/` when the training is completed.

## Test trained SelfReg ResNet18 model
To test a ResNet18, you can download SelfReg model below.
| Backbone        | Target Domain |Acc %            | models |
| :--------------:| :-----------: | :------------:  |:------------: |
| ResNet-18       |Photo          |96.83            |[download]()   |
| ResNet-18       |Art         |83.15            |[download]()   |
| ResNet-18       |Cartoon        |79.61            |[download]()   |
| ResNet-18       |Sketch            |78.90            |[download]()  |

