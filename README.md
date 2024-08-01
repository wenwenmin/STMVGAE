# STMVGAE:Deep Clustering Representation for Spatially Resolved Transcriptomics Data via Multi-view Variational Graph Auto-Encoders with Consensus Clustering
![image](https://github.com/JinyunNiu/STMVGAE/blob/main/STMVGAE_Overview.jpg)

## Overview
The rapid development of spatial transcriptomics technology has provided unprecedented opportunities to understand tissue relationships and functions within specific spatial contexts. Accurate identification of spatial domains is crucial for downstream spatial transcriptomics analysis, which lies in making full use of the spatial background of tissue and the integration of cell gene expression. However, effectively integrating gene expression data with spatial location information to identify spatial domains remains a challenge. To deal with the above issue, we propose STMVGAE, an accurate and general deep learning framework to identify spatial domains. 
Specifically, we extract the histological image feature through a pre-trained convolutional neural network (CNN) and fuse them with gene expression data to obtain multi-modal gene expression data. Additionally, to leverage spatial location information fully, we employ various construction methods to create multiple graphs (views) with different similarities. STMVGAE then learns multiple low-dimensional latent embeddings by combining multiple views and multi-modal data using variational autoencoders with graph convolutions (VGAE). We cluster the multiple low-dimensional latent embeddings and introduce consensus clustering to integrate the clustering results. To our knowledge, STMVGAE is the first method to employ consensus clustering for integrating multi-view clustering results. Consensus clustering has been demonstrated to effectively enhance the stability and robustness of the outcomes. 
We conduct full experiments on STMVGAE on five real datasets and compared it with five state-of-the-art methods, STMVGAE consistently achieves competitive results across all datasets. We assess not only spatial domain identification capabilities of STMVGAE but also its performance in Umap visualization, trajectory inference, spatial variant genes (SVGs) identification, denoising, batch integration, and other downstream tasks. In summary, our results demonstrate that STMVGAE excels in identifying spatial domains while performing a range of downstream tasks, making it an invaluable tool for uncovering new insights in spatial transcriptomics research.

## File description
* adj: We construct a variety of adjacency matrices with different degrees of similarity through gene expression data and spatial location information to train STMVGAE, allowing for a variety of parameter selections. Data processing for graph neural networks is also included.   
* his_feat: We use the trained CNN to extract morphological image features and enhance them. In this file, we set multiple optional parameters, and users can choose a variety of CNNs for training.   
* augument: In this part, we fuse the preprocessed gene expression and morphological image features. Similarly, there are a variety of linear rectification units to choose from in the fusion part.
* dataset: Data reading and data preprocessing.
* model: STMVGAE model structure.
* utils: Data processing and clustering methods.
* train: Training STMVGAE.
* consensus_clustering: Consensus clustering to integrate diverse results.
* SVGs: Spatially variant gene (SVGs) identification.

## System environment
scanpy == 1.9.2   
scipy == 1.10.1   
sklearn == 1.3.1   
PyG == 2.3.1

## Datasets
All publicly available ST datasets, used in this study, can be downloaded from https://zenodo.org/records/13119867 or find them on the following websites：
-  10x Visium human dorsolateral prefrontal cortex dataset: http://spatial.libd.org/spatialLIBD/;
-  10x Visium human breast cancer dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0;
-  10x Visium Human breast cancer: ductal carcinoma in situ dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0; https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0; 
-  sptial-research melanoma cancer dataset: https://github.com/1alnoman/ScribbleDom;
-  Stereo-seq mouse olfactory bulb dataset: https://github.com/JinmiaoChenLab/SEDR_analyses;
-  The ISH images of the adult human brain in gene denosing task are available at the Allen Human Brain Atlas: \url{https://human.brain-map.org/}.

## Getting started
We provide slices of the DLPFC dataset. We have encapsulated the STMVGAE training process into the run.ipynb file. You only need to enter the correct file path to run it.
If you need programs to run other downstream tasks, please contact us via email: 83024551@qq.com.
- See `run.ipynb`
