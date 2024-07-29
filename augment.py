import math
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from tqdm import tqdm

def cal_weight_matrix(
		adata,
		md_dist_type="cosine",
		):
	morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
	morphological_similarity[morphological_similarity < 0] = 0
	print("Morphological similarity calculting Done!")
	adata.obsm["morphological_similarity"] = morphological_similarity
	return adata

def find_adjacent_spot(
	adata,
	use_data = "raw",
	neighbour_k = 3,
	verbose = False,
	):

	#把adata.X取出来
	if use_data == "raw":
		if isinstance(adata.X, csr_matrix):
			gene_matrix = adata.X.toarray()
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		gene_matrix = adata.obsm[use_data]
	weights_list = []
	final_coordinates = []
	with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
		#将上面函数得到的weights_matrix_all提取出来，1.筛选出符合要求的k个邻居，并取出对应的权重。2.权重做比值spot_weight / spot_weight.sum()，然后用比值*adata.X对应的行。
		#3.求和，保存。
		for i in range(adata.shape[0]):
			# 将adata.obsm['weights_matrix_all']的每一行进行排序,adata.obsm['weights_matrix_all'][i]取出第i行
			# argsort会按照从小到的返回对应位置的index，
			# [-neighbour_k:]返回最后四位，[-neighbour_k:][:neighbour_k-1]返回交集中的几位(3位)
			# 求出一个（1*3）矩阵保存到current_spot，排序中除最后一个外较大的3个元素
			current_spot = adata.obsm['morphological_similarity'][i].argsort()[-neighbour_k:][:neighbour_k-1] #取出值最大的三个点
			# 将这3个元素对应的值取出来
			spot_weight = adata.obsm['morphological_similarity'][i][current_spot] ##取出值最大的三个点对应的值
			#在gene_matrix中（adata.X）中将这三行取出来
			spot_matrix = gene_matrix[current_spot]
			if spot_weight.sum() > 0:
				spot_weight_scaled = (spot_weight / spot_weight.sum()) #求spot_weight占比
				weights_list.append(spot_weight_scaled)
				spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)#按位相乘,将占比与提取出来的3个adata.X相乘
				spot_matrix_final = np.sum(spot_matrix_scaled, axis=0) #3行按行求和变为一行
			else:
				spot_matrix_final = np.zeros(gene_matrix.shape[1])
				weights_list.append(np.zeros(len(current_spot)))
			final_coordinates.append(spot_matrix_final)
			pbar.update(1)
		adata.obsm['adjacent_data'] = np.array(final_coordinates)
		if verbose:
			adata.obsm['adjacent_weight'] = np.array(weights_list)
		return adata

def augment_gene_data(
	adata,
	adjacent_weight = 0.2,
	):
	if isinstance(adata.X, np.ndarray):
		augement_gene_matrix =  adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	elif isinstance(adata.X, csr_matrix):
		augement_gene_matrix = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	adata.obsm["augment_gene_data"] = augement_gene_matrix
	return adata

def augment_adata(
	adata,
	md_dist_type="cosine",
	use_data = "raw",
	neighbour_k = 3,
	adjacent_weight = 0.2,
	):
	adata = cal_weight_matrix(
				adata,
				md_dist_type = md_dist_type
				)
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k)
	adata = augment_gene_data(adata,
				adjacent_weight = adjacent_weight)
	return adata

def augment_adata_(adata,
				   activate='Relu',
				   min_cells = 50,
	               min_counts = None,
				   weight = 0.1,
				   ):
	print('Augment adata is processing')
	data = adata.obsm['image_feat'].astype(np.float64)
	data = torch.FloatTensor(data)
	conv1 = torch.nn.Linear(data.shape[1], 3000)
	data1 = conv1(data)
	data2 = torch.nn.BatchNorm1d(data1.shape[1])(data1)
	if activate == 'Sigmoid': #softmax后基本都是0,1，密度比softmax大
		sigmoid = torch.nn.Sigmoid()
		data3 = sigmoid(data2)
		data3 = data3.detach().numpy()
	if activate == 'Softmax': #softmax后基本都是0,1，且0很多
		softmax = torch.nn.Softmax()
		data3 = softmax(data2)
		data3 = data3.detach().numpy()
	if activate == 'Relu':
		relu = torch.nn.ReLU()
		data3 = relu(data2)
		data3 = data3.detach().numpy()
	if activate == 'Relu6':
		relu6 = torch.nn.ReLU6()
		data3 = relu6(data2)
		data3 = data3.detach().numpy()
	data4 = np.around(weight * data3)

	sc.pp.calculate_qc_metrics(adata, inplace=True)  # qc质量测试
	sc.pp.filter_genes(adata, min_cells=min_cells)  # 基因数量筛选
	if min_counts is not None:
		sc.pp.filter_cells(adata, min_counts=min_counts)
	print("After filtering: ", adata.shape)  # 输出filtering后基因的shape
	sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)  # 筛选高表达基因3000
	adata = adata[:, adata.var['highly_variable']]

	augument_data = adata.X.toarray() + data4
	adata.obsm['augument_data'] = augument_data

	adata.X = adata.obsm["augument_data"].astype(np.float64)
	sc.pp.normalize_total(adata)
	sc.pp.log1p(adata)
	print('Augment adata is ending')

	return adata







