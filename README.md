## STMVGAE: Deep Clustering Representation for Spatially Resolved Transcriptomics Data via Multi-view Variational Graph Auto-Encoders with Consensus Clustering
![image](https://github.com/JinyunNiu/STMVGAE/blob/main/STMVGAE_Overview.jpg)

## Overview
The rapid development of spatial resolved transcriptomics (SRT) has provided unprecedented opportunities to understand tissue relationships and functions within specific spatial contexts. Accurate identification of spatial domains is crucial for downstream analysis, which lies in making full use of the gene expression profiles, spatial location information and histological images provided by SRT data. 
However, integrating the diverse data types provided by SRT, such as gene expression profiles, spatial location and histological images, poses significant challenges due to the inherent complexity, noise, and sparsity of the data. Single-view models, which rely on one type of spatial graph, often fail to fully capture the intricate spatial structures and relationships within the tissue. 
To this end, we propose STMVGAE, a consensus clustering framework that uses a multi-view strategy to accurately identify spatial domains. Specifically, STMVGAE extracts the histological image feature through a pre-trained convolutional neural network (CNN) and fuses them with gene expression profiles data to obtain augmented gene expression profiles data. Next, to fully leverage spatial location information, STMVGAE employs various construction methods to create multiple graphs (views) with different similarities. Subsequently, STMVAGE takes different views and enhanced gene expression profiles as input to a variational graph autoencoders (VGAEs) and trains multiple view-specific embeddings for clustering. Finally, STMVGAE uses a consensus clustering strategy to integrate view-specific clustering outputs into the final consensus clustering labels for spatial domain identification.
We apply STMVGAE on five real datasets and compare it with five state-of-the-art methods, STMVGAE consistently achieves competitive results across all datasets. We assess not only spatial domain identification capabilities of STMVGAE but also its performance in Umap visualization, Paga trajectory inference, spatial variant genes (SVGs) identification, denoising, batch integration, and other downstream tasks. In summary, our results demonstrate that STMVGAE excels in identifying spatial domains while performing a range of downstream tasks, making it an invaluable tool for uncovering new insights in spatial transcriptomics research.

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
-  The ISH images of the adult human brain in gene denosing task are available at the Allen Human Brain Atlas: https://human.brain-map.org/.

## Contact details
If you have any questions, please contact：niujinyun@aliyun.com.

## Citing
<p>The corresponding BiBTeX citation are given below:</p>
<div class="highlight-none"><div class="highlight"><pre>
@article{niu2024STMVGAE,
  title={Deep clustering representation of spatially resolved transcriptomics data using multi-view variational graph auto-encoders with consensus clustering},
  author={Niu, Jinyun and Zhu, Fangfang and Xu, Taosheng and Wang, Shunfang and Min, Wenwen},
  journal={Computational and Structural Biotechnology Journal},
  volume={23},
  pages={4369--4383},
  year={2024},
  publisher={Elsevier}
}
</pre></div>
