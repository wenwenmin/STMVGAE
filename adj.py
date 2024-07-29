import os,sys
import numpy as np
import torch
import sklearn
import scipy
import scanpy as sc
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from scipy.spatial import distance
from torch_sparse import SparseTensor
from scipy import stats

class graph():
    def __init__(self,
                 ann_data,
                 distType='Radius_balltree',
                 rad_cutoff=250,
                 verbose=True,
                 max_neigh=50,
                 k_cutoff=12,
                 ):
        self.ann_data = ann_data
        self.distType = distType
        self.rad_cutoff = rad_cutoff
        self.verbose = verbose
        self.max_neigh = max_neigh
        self.k_cutoff = k_cutoff

    def compute_spatial_net(self):

        if self.verbose:
            print('------Calculating spatial graph...')
        coor = pd.DataFrame(self.ann_data.obsm['spatial'])
        coor.index = self.ann_data.obs.index
        coor.columns = ['imagerow', 'imagecol']

        # 参数有max_neigh，rad_cutoff，distType
        if self.distType == 'Radius_balltree':
            from sklearn.neighbors import NearestNeighbors
            nbrs = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.max_neigh + 1, algorithm='ball_tree').fit(coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
            distances, indices = nbrs.kneighbors(coor) #返回两个节点数*（max_neigh+1）的矩阵，distances返回i节点和最近的50个节点的距离，indices返回i节点和最近的50个节点的索引
            indices = indices[:, 1:] #删除i自身
            distances = distances[:, 1:]#删除i自身
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
            KNN_df = pd.concat(KNN_list)
            KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
            Spatial_Net1 = KNN_df.loc[KNN_df['Distance'] < self.rad_cutoff,]  # 筛选150以内的点
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)

        # 参数有max_neigh，k_cutoff，distType
        elif self.distType == 'KNN_balltree':
            from sklearn.neighbors import NearestNeighbors
            nbrs = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.max_neigh + 1, algorithm='ball_tree').fit(coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
            distances, indices = nbrs.kneighbors(coor)
            indices = indices[:, 1:self.k_cutoff + 1]
            distances = distances[:, 1:self.k_cutoff + 1]
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
            KNN_df = pd.concat(KNN_list)
            KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
            Spatial_Net1 = KNN_df.copy()
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)

        elif self.distType == 'Radius_kdtree':
            from sklearn.neighbors import NearestNeighbors
            nbrs = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.max_neigh + 1, algorithm='kd_tree').fit(coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
            distances, indices = nbrs.kneighbors(coor)
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
            KNN_df = pd.concat(KNN_list)
            KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
            Spatial_Net1 = KNN_df.loc[KNN_df['Distance'] < self.rad_cutoff,]  # 筛选150以内的点
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)


        elif self.distType == 'KNN_kdtree':
            from sklearn.neighbors import NearestNeighbors
            nbrs = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.max_neigh + 1, algorithm='kd_tree').fit(coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
            distances, indices = nbrs.kneighbors(coor)
            indices = indices[:, 1:self.k_cutoff + 1]
            distances = distances[:, 1:self.k_cutoff + 1]
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
            KNN_df = pd.concat(KNN_list)
            KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
            Spatial_Net1 = KNN_df.copy()
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)

        elif self.distType == 'Brute':
            from sklearn.neighbors import NearestNeighbors
            nbrs = sklearn.neighbors.NearestNeighbors(
                n_neighbors=self.max_neigh + 1, algorithm='brute').fit(coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
            distances, indices = nbrs.kneighbors(coor)
            indices = indices[:, 1:self.k_cutoff + 1]
            distances = distances[:, 1:self.k_cutoff + 1]
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
            KNN_df = pd.concat(KNN_list)
            KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
            Spatial_Net1 = KNN_df.copy()
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)

        elif self.distType == 'Genes_cosine':
            from sklearn.metrics import pairwise_distances
            min_cells = 50
            min_counts = None

            data = self.ann_data.copy()
            sc.pp.calculate_qc_metrics(data, inplace=True)  # qc质量测试
            sc.pp.filter_genes(data, min_cells=min_cells)  # 基因数量筛选
            if min_counts is not None:
                sc.pp.filter_cells(data, min_counts=min_counts)
            #print("After filtering: ", data.shape)  # 输出filtering后基因的shape
            sc.pp.highly_variable_genes(data, flavor='seurat_v3', n_top_genes=3000)  # 筛选高表达基因3000
            data = data[:, data.var['highly_variable']]
            sc.pp.pca(data, n_comps=50)

            gene_correlation = 1 - pairwise_distances(data.obsm['X_pca'], metric='cosine')

            for i in range(gene_correlation.shape[0]):
                gene_correlation[i][i] = 0

            KNN_DataFrame = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
            for node_idx in range(gene_correlation.shape[0]):  # 遍历所有节点
                tmp = gene_correlation[node_idx, :].reshape(1, -1)
                indices = tmp.argsort()[0][-(self.k_cutoff):].reshape(1, -1)[0]
                distances = tmp[:, indices][0]
                for i in range(self.k_cutoff):
                    KNN_DataFrame = KNN_DataFrame.append(pd.Series([node_idx, indices[i], distances[i]],
                                                                   index=['Cell1', 'Cell2', 'Distance']),ignore_index=True)

            Spatial_Net1 = KNN_DataFrame.copy()
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)

        elif self.distType == 'Spearmanr':
            from scipy import stats

            min_cells = 50
            min_counts = None

            data = self.ann_data.copy()
            sc.pp.calculate_qc_metrics(data, inplace=True)  # qc质量测试
            sc.pp.filter_genes(data, min_cells=min_cells)  # 基因数量筛选
            if min_counts is not None:
                sc.pp.filter_cells(data, min_counts=min_counts)
            # print("After filtering: ", data.shape)  # 输出filtering后基因的shape
            sc.pp.highly_variable_genes(data, flavor='seurat_v3', n_top_genes=3000)  # 筛选高表达基因3000
            data = data[:, data.var['highly_variable']]
            sc.pp.pca(data, n_comps=50)

            SpearA, _ = stats.spearmanr(data.obsm['X_pca'], axis=1)

            for i in range(SpearA.shape[0]):
                SpearA[i][i] = 0

            KNN_DataFrame = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
            for node_idx in range(SpearA.shape[0]):  # 遍历所有节点
                tmp = SpearA[node_idx, :].reshape(1, -1)
                indices = tmp.argsort()[0][-(self.k_cutoff):].reshape(1, -1)[0]
                distances = tmp[:, indices][0]
                for i in range(self.k_cutoff):
                    KNN_DataFrame = KNN_DataFrame.append(pd.Series([node_idx, indices[i], distances[i]],
                                                                   index=['Cell1', 'Cell2', 'Distance']),
                                                         ignore_index=True)

            Spatial_Net1 = KNN_DataFrame.copy()
            id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
            cell1, cell2 = Spatial_Net1['Cell1'].map(id_cell_trans), Spatial_Net1['Cell2'].map(id_cell_trans)
            Spatial_Net = Spatial_Net1.assign(Cell1=cell1, Cell2=cell2)


        else:
            raise ValueError('Unknown disType')

        if self.verbose:
            print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], self.ann_data.n_obs))
            print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / self.ann_data.n_obs))
        self.ann_data.uns['Spatial_Net'] = Spatial_Net
        self.ann_data.uns['Spatial_Net1'] = Spatial_Net1

    def compute_edge_list(self):
        G_df = self.ann_data.uns['Spatial_Net'].copy()
        cells = np.array(self.ann_data.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
        G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
                          shape=(self.ann_data.n_obs, self.ann_data.n_obs))
        G = G + scipy.sparse.eye(G.shape[0])
        edge_list = np.nonzero(G)
        return edge_list


    def Stats_Spatial_Net(self):
        import matplotlib.pyplot as plt
        Num_edge = self.ann_data.uns['Spatial_Net']['Cell1'].shape[0]  # 获取边数量
        Mean_edge = Num_edge / self.ann_data.shape[0]  # 求均值
        # 统计adata.uns['Spatial_Net']['Cell1']中每个细胞作为起点的边的数量
        plot_df = pd.value_counts(pd.value_counts(self.ann_data.uns['Spatial_Net']['Cell1']))
        # 并再次使用pd.value_counts对这些数量进行统计
        plot_df = plot_df / self.ann_data.shape[0]
        fig, ax = plt.subplots(figsize=[3, 2])
        plt.ylabel('Percentage')
        plt.xlabel('')
        plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
        ax.bar(plot_df.index, plot_df)


    def compute_edge_dict_(self):
        df = self.ann_data.uns['Spatial_Net1'].copy()
        df1 = df.copy()
        df1 = np.array(df1)[:,:2].tolist()
        df2 =[]
        for [i, j] in df1:
            tmp = (int(i), int(j))
            df2.append(tmp)
        graphList = df2
        return graphList


    def List2Dict(self, graphList):
        """
        Return dict: eg {0: [0, 3542, 2329, 1059, 397, 2121, 485, 3099, 904, 3602],
                     1: [1, 692, 2334, 1617, 1502, 1885, 3106, 586, 3363, 101],
                     2: [2, 1849, 3024, 2280, 580, 1714, 3311, 255, 993, 2629],...}
        """
        #graphList为N*k维矩阵，N为节点数，K为邻居数
        graphdict = {}
        tdict = {}
        for graph in graphList:
            end1 = graph[0] #节点数
            end2 = graph[1] #邻居数（k）
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.ann_data.shape[0]):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def mx2SparseTensor(self, mx):

        """Convert a scipy sparse matrix to a torch SparseTensor.
            将mx(密集矩阵)转换为torch支持的类型
        """
        mx = mx.tocoo().astype(np.float32)  # 将密集矩阵转化为稀疏矩阵
        row = torch.from_numpy(mx.row).to(torch.long)  # 取出行
        col = torch.from_numpy(mx.col).to(torch.long)  # 取出列
        values = torch.from_numpy(mx.data)  # 取出值
        adj = SparseTensor(row=row, col=col,
                           value=values, sparse_sizes=mx.shape)  # 转为torch.SparseTensor
        adj_ = adj.t()
        return adj_

    # 对adj进行预处理，自环以及归一化
    def pre_graph(self, adj):

        """ Graph preprocessing.
            D^(-1/2)*A*D^(-1/2)
        """
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])  # 添加自环,对角线加1
        rowsum = np.array(adj_.sum(1))  # 计算每一行sum
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 每一行之和先取（-1/2），flatten()变为一行，然后构建一个对角矩阵
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # 归一化

        return self.mx2SparseTensor(adj_normalized)  # 将归一化以后的矩阵转成torch.SparseTensor

    def main(self):
        adj_mtx = self.compute_spatial_net()  # 得到图的邻接矩阵
        graphlist = self.compute_edge_dict_()
        graphdict = self.List2Dict(graphlist)  # 转换为Dict
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))  # 转换为 NetworkX 图对象,密集矩阵4226*4226

        """ Store original adjacency matrix (without diagonal entries) for later 
            adj_pre.diagonal()返回对角线元素

        """
        adj_pre = adj_org
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)  # 去除自环
        adj_pre.eliminate_zeros()  # 稀疏化处理，稀疏矩阵中不会显示值为0的部分

        """ Some preprocessing."""
        adj_norm = self.pre_graph(adj_pre)  # 归一化后的稀疏邻接矩阵，类型为SparseTensor
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])  # 带有对角线元素的邻接矩阵，用作图自动编码器的标签
        adj_label = torch.FloatTensor(adj_label.toarray())
        # n_node=4226,norm=4226*4226/
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)

        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm}

        return graph_dict

def combine_graph_dict(dict_1, dict_2):
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())

    graph_dict = {
        "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
        "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])}
    return graph_dict


