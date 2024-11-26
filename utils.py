import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score,homogeneity_score,contingency_matrix

import torch
from torch_geometric.data import Data

def Transfer_pytorch_Data(ann_data):
    ''''''
    G_df = ann_data.uns['Spatial_Net'].copy()
    cells = np.array(ann_data.obs_names) #通过obs获取barcodes序列，转化为np.array，shape->(4226，)
    cells_id_tran = dict(zip(cells, range(cells.shape[0]))) #打包为一个dic，keys为细胞名，values为其索引
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran) #进行map映射
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    #构建稀疏矩阵,sp.coo_matrix(data,(row,col),shape(N*N)),有连接的置为1，shape->(16904,16904)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(ann_data.n_obs, ann_data.n_obs))
    G = G + sp.eye(G.shape[0]) #添加自环,对角线全为1

    edgeList = np.nonzero(G) ##可以把G中非0的元素提取出来构造edgeList


    # 转为可训练的Tensor
    if type(ann_data.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(ann_data.X),
            ) # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(ann_data.X.todense()),
            ) # .todense()
    return data


def mclust_R(ann_data, num_cluster=7, modelNames='EEE', used_obsm='VGAE_GCN_DEC_Image', random_seed=0):
    """\
    该函数用于在Python中使用mclust算法对给定数据进行聚类，并将聚类结果保存在AnnData对象的obs中。
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(ann_data.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    ann_data.obs['mclust'] = mclust_res
    ann_data.obs['mclust'] = ann_data.obs['mclust'].astype('int')
    ann_data.obs['mclust'] = ann_data.obs['mclust'].astype('category')
    return ann_data

def KMeans_P(ann_data, n_clusters, use_rep='h', init='k-means++', n_init=10, max_iter=300, random_state=0):
    data = ann_data.obsm[use_rep]
    kmeans = KMeans(n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state).fit_predict(data)

    ann_data.obs['kmeans'] = kmeans
    ann_data.obs['kmeans'] = ann_data.obs['kmeans'].astype('int')
    ann_data.obs['kmeans'] = ann_data.obs['kmeans'].astype('category')

    return ann_data


def find_res(ann_data, use_rep='h', num_cluster=7, resolution: list = list(np.arange(0.1, 1.5, 0.01))):
    print("Finding best ari,please wait a moment")
    score = []
    for r in resolution:
        sc.pp.neighbors(ann_data, use_rep=use_rep)
        sc.tl.louvain(ann_data, resolution=r)
        pred_num_cluster = len(ann_data.obs['louvain'].unique())
        if pred_num_cluster == num_cluster:
            indices = np.logical_not(ann_data.obs["Manual annotation"].isna())
            ground_truth = ann_data.obs["Manual annotation"].dropna()
            ari = adjusted_rand_score(ann_data.obs['louvain'][indices], ground_truth[indices])
            score.append(ari)
        else:
            score.append(0)
    best_ari = np.max(score)
    print("Best ari is: {:.4f}".format(best_ari))

    return best_ari

def loss_fig(ann_data,epoch):
    epochs = np.arange(epoch)
    loss_list = ann_data.uns['loss']
    loss_list = np.array(torch.tensor(loss_list, device='cpu'))
    plt.figure(figsize=(5, 5), dpi=80)
    plt.title('loss')
    plt.plot(epochs, loss_list, label='train_loss')
    plt.legend()
    plt.show()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'Method', "DLPFC"])

    ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, ari, methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, nmi, methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    hs = homogeneity_score(pred, gt)
    df = df.append(pd.Series([sample, hs, methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, purity, methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    return df

def refine(
    sample_id,
    pred,
    dis,
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

