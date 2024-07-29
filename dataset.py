import torch
import os
import torch.nn as nn
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

def read_10X_Visium(path = "../dataset/DLPFC/",
                    #path='../dataset/BRCA1',
                    #path='../dataset/bcdc_ffpe',
                    section_id = "151507",
                    genome = None,
                    library_id = None,
                    load_images =True,
                    quality ='hires',
                    image_path = None):
    adata = sc.read_visium(
                          #os.path.join(path,'reading_h5'),
                          #os.path.join(path,'V1_Human_Breast_Cancer_Block_A_Section_1'),
                          os.path.join(path, section_id),
                          count_file= section_id+'_filtered_feature_bc_matrix.h5',
                          #count_file='bcdc_ffpe_filtered_feature_bc_matrix.h5',
                          genome = genome,
                          library_id = library_id,
                          load_images = load_images,
                          )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


#选取高变基因作为输入
def data_process(ann_data,min_cells=50,min_counts=None,flavor="seurat_v3",n_top_genes=3000):
    #ann_data.layers['count'] = ann_data.X
    sc.pp.calculate_qc_metrics(ann_data, inplace=True)  # qc质量测试
    sc.pp.filter_genes(ann_data, min_cells=min_cells)  # 基因数量筛选
    if min_counts is not None:
        sc.pp.filter_cells(ann_data, min_counts=min_counts)
    print("After filtering: ", ann_data.shape)  # 输出filtering后基因的shape
    sc.pp.highly_variable_genes(ann_data, flavor=flavor, n_top_genes=n_top_genes)  # 筛选高表达基因3000
    ann_data = ann_data[:, ann_data.var['highly_variable']]
    sc.pp.normalize_total(ann_data, target_sum=1e4)  # normalize
    sc.pp.log1p(ann_data)

    return ann_data

#pca降维作为输入
def data_process_pca(ann_data,min_cells=50,min_counts=None,pca_n_comps=1000,use_morphological=True):
    if use_morphological==True:
        if 'augment_gene_data' not in ann_data.obsm.keys():
            raise ValueError("augment_gene_data is not existed! Run augment_adata first!")
        else:
            ann_data.X = ann_data.obsm["augment_gene_data"].astype(np.float64)
    else:
        ann_data.X = ann_data.X
    #ann_data.X = ann_data.obsm["augment_gene_data"].astype(np.float64)
    data = sc.pp.normalize_total(ann_data, target_sum=1, inplace=False)['X']
    data = sc.pp.log1p(data)
    data = sc.pp.scale(data)
    data = sc.pp.pca(data, n_comps=pca_n_comps)
    return data

