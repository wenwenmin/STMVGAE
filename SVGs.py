import os
import numpy as np
import scanpy as sc
import SpaGCN as spg
import matplotlib.colors as clr

from scipy.sparse import issparse

DLPFC_dir = ''
section_id = ''
adata = sc.read_visium('') #trained file address

raw = sc.read_visium(os.path.join(DLPFC_dir, section_id),
                          count_file=section_id + '_filtered_feature_bc_matrix.h5')
raw.var_names_make_unique()
raw.obs["pred"]=adata.obs["consensus_label"].astype('category')
raw.obs["x_array"]=raw.obs["array_row"]
raw.obs["y_array"]=raw.obs["array_col"]
raw.obs["x_pixel"] = raw.obsm['spatial'][:,0]
raw.obs["y_pixel"] = raw.obsm['spatial'][:,1]
x_array = raw.obs["x_array"].tolist()
y_array = raw.obs["y_array"].tolist()
x_pixel = raw.obs["x_pixel"].tolist()
y_pixel = raw.obs["y_pixel"].tolist()
raw.X = (raw.X.A if issparse(raw.X) else raw.X)
raw.raw = raw
sc.pp.log1p(raw)

#Use domain 0 as an example
target=1
#Set filtering criterials
min_in_group_fraction=0.8
min_in_out_group_ratio=1
min_fold_change=1
#Search radius such that each spot in the target domain has approximately 10 neighbors on average
adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
r=spg.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs["consensus_label"].tolist(), start=start, end=end, num_min=10, num_max=200,  max_run=100)
#Detect neighboring domains
nbr_domians=spg.find_neighbor_clusters(target_cluster=target,
                                   cell_id=raw.obs.index.tolist(),
                                   x=raw.obs["x_array"].tolist(),
                                   y=raw.obs["y_array"].tolist(),
                                   pred=raw.obs["pred"].tolist(),
                                   radius=r,
                                   ratio=1/2)
nbr_domians=nbr_domians[0:3]
de_genes_info=spg.rank_genes_groups(input_adata=raw,
                                target_cluster=target,
                                nbr_list=nbr_domians,
                                label_col="pred",
                                adj_nbr=True,
                                log=True)
#Filter genes
de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
filtered_info=de_genes_info
filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
                            (filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
                            (filtered_info["in_group_fraction"]>min_in_group_fraction) &
                            (filtered_info["fold_change"]>min_fold_change)]
filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
filtered_info["target_dmain"]=target
filtered_info["neighbors"]=str(nbr_domians)
print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())

gene_list = ['ENC1'] #

color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
for g in gene_list:
#for g in filtered_info["genes"].tolist():
    raw.obs["exp"]=raw.X[:,raw.var.index==g]
    sc.pl.spatial(raw, img_key="hires",
              color="exp",
              title=g,
              color_map=color_self)

target=1
meta_name, meta_exp=spg.find_meta_gene(input_adata=raw,
                    pred=raw.obs["pred"].tolist(),
                    target_domain=target,
                    start_gene="ENC1",
                    mean_diff=0,
                    early_stop=True,
                    max_iter=3,
                    use_raw=False)

raw.obs["meta"]=meta_exp
raw.obs["exp"]=raw.obs["meta"]
sc.pl.spatial(raw, img_key="hires",
              color="exp",
              title='meta',
              color_map=color_self)
