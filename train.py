import time

from tqdm import tqdm

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from pathlib import Path

from model import VGAE
from utils import *
from adj import graph
from dataset import *
from his_feat import *
from augment import augment_adata,augment_adata_



class train_model():
    def __init__(self,
                 input_dim,
                 path="../dataset/DLPFC/",
                 section_id='151507',
                 cnnType='ResNet50',
                 activate='Relu',
                 distType ='Genes_cosine',
                 rad_cutoff=250,
                 k_cutoff=11,
                 pca_components=50,
                 weight = 0.5,
                 pre_epochs=1000,
                 epochs=500,
                 lr=1e-3,
                 weight_decay=1e-4,
                 random_seed=0,
                 mse_weight=1,
                 bce_kld_weight=0.1,
                 kl_weight=1,
                 State_Net=False,
                 use_gpu=True,
                 ):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.ann_data = read_10X_Visium(path=path,section_id=section_id)
        self.ann_data = get_image_crop(self.ann_data,section_id=section_id,cnnType=cnnType,pca_n_comps=pca_components)
        self.ann_data = augment_adata_(self.ann_data,activate=activate,weight=weight)
        self.X = self.ann_data.X
        self.X = torch.FloatTensor(self.X).to(self.device)
        self.net = graph(self.ann_data,distType=distType,rad_cutoff=rad_cutoff,k_cutoff=k_cutoff)
        self.net.compute_spatial_net()
        if State_Net == True:
            self.net.Stats_Spatial_Net()
        self.graph_dict = self.net.main()
        self.adj_norm = self.graph_dict['adj_norm'].to(self.device)
        self.adj_label = self.graph_dict['adj_label'].to(self.device)
        self.norm = self.graph_dict['norm_value']
        self.data = Transfer_pytorch_Data(self.ann_data).to(self.device)
        self.model = VGAE(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.lr = lr
        self.dec_tol = 0
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.num_spots = self.ann_data.shape[0]
        self.q_stride = 20
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.kl_weight = kl_weight


    def pre_train(
            self,
            gradient_clipping=5.,
    ):
        seed_everything(random_seed=0)

        loss_list = []
        start_time = time.time()
        print('train an initial model：')
        for epoch in tqdm(range(self.pre_epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            _, _, mu, logvar, z, rec_adj, rec_x, _, _ = self.model(self.X, self.data.edge_index)

            mes_loss = F.mse_loss(rec_x, self.X)
            bce_loss = self.norm * F.binary_cross_entropy(rec_adj, self.adj_label)
            KLD = -0.5 / self.num_spots * torch.mean(torch.sum(
                1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
            loss =self.mse_weight * mes_loss + self.bce_kld_weight * (bce_loss+KLD)
            loss_list.append(loss)
            loss.backward()
            # 梯度截断
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
            self.optimizer.step()
        end_time = time.time()
        print('Elapsed training time:{:.4f} seconds'.format((end_time - start_time)))

        '''
        self.model.eval()
        h, conv, mu, logvar, z, rec_adj, rec_x, z_h, q = self.model(self.data.x, self.data.edge_index)
    
        h_rep = h.to('cpu').detach().numpy()
        self.ann_data.obsm['h'] = h_rep
        z_rep = z.to('cpu').detach().numpy()
        self.ann_data.obsm['z'] = z_rep
        z_h_rep = z_h.to('cpu').detach().numpy()
        self.ann_data.obsm['z_h'] = z_h_rep
    
        return self.ann_data
        '''


    @torch.no_grad()
    def process(
            self,
    ):
        self.model.eval()
        h, conv, mu, logvar, z, rec_adj, rec_x, z_h, q = self.model(self.X, self.data.edge_index)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    def save_model(
        self,
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self,
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def fit(self,
            cluster_n=20,
            clusterType='Louvain',
            res=1.0,
            pretrain=True,
            ):
        seed_everything(random_seed=2023)
        if pretrain:
            self.pre_train()
            z, _ = self.process()
        if clusterType == 'KMeans':
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=2023) #定义Kmeans
            y_pred_last = np.copy(cluster_method.fit_predict(z)) #预测将所有样本分到cluster_n类中
            self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device) #通过.cluster_centers_获取所有的聚类中心，聚类中心的维度为cluster_n*8
        elif clusterType == 'Louvain':
            cluster_data = sc.AnnData(z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution=res)
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy() #得到louvain聚类的结果
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(z, index=np.arange(0, z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean()) #属于一类的全部分到一起求平均
            self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
        elif clusterType == 'Mclust':
            cluster_data = sc.AnnData(z)
            cluster_data.obsm['z'] = z
            mclust_R(cluster_data, used_obsm='z', num_cluster=cluster_n)
            y_pred_last = cluster_data.obs['mclust'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(z, index=np.arange(0, z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)

        print('train an final model:')
        for epoch in tqdm(range(self.epochs)):
            if epoch % self.q_stride == 0:
                _, q = self.process()
                q = self.model.target_distribution(torch.Tensor(q).clone().detach())  # 软分布
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()  # 预测结果，argmax（）找到最大值，当前样本最有可能属于的簇
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch > 0 and delta_label < self.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            torch.set_grad_enabled(True)
            self.model.train()
            self.optimizer.zero_grad()
            h, conv, mu, logvar, z, rec_adj, rec_x, z_h, out_q = self.model(self.X, self.data.edge_index)
            mes_loss = F.mse_loss(rec_x, self.X)
            bce_loss = self.norm * F.binary_cross_entropy(rec_adj, self.adj_label)
            KLD = -0.5 / self.num_spots * torch.mean(torch.sum(
                1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
            kl_loss = F.kl_div(out_q.log(), q.to(self.device))
            loss = self.mse_weight * mes_loss + self.bce_kld_weight * (bce_loss+KLD)  + self.kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        self.model.eval()
        h, conv, mu, logvar, z, rec_adj, rec_x, z_h, out_q = self.model(self.X, self.data.edge_index)

        z_rep = z.to('cpu').detach().numpy()
        self.ann_data.obsm['z'] = z_rep
        recon_rep = rec_x.to('cpu').detach().numpy()
        self.ann_data.obsm['recon_x'] = recon_rep

        return self.ann_data


def kld_loss_function(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

    loss = kld(p, q)

    return loss


def seed_everything(random_seed):
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)