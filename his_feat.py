import os
import math
import anndata
import numpy as np 
import scanpy as sc
import pandas as pd 
from PIL import Image
from pathlib import Path
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torchvision.transforms as transforms

#该类用于提取图像特征，并将结果保存到 adata.obsm["image_feat_pca"] 中。用于做后续的数据增强
class image_feature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='ResNet50',
        verbose=False,
        seeds=0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(
        self,
        ):

        if self.cnnType == 'ResNet18':
            cnn_pretrained_model = models.resnet18(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'ResNet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Shufflenet':
            cnn_pretrained_model = models.shufflenet_v2_x2_0(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Squeezenet':
            cnn_pretrained_model = models.squeezenet1_1(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_t':
            cnn_pretrained_model = models.swin_t(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_b':
            cnn_pretrained_model = models.swin_b(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_s':
            cnn_pretrained_model = models.swin_s(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_v2_t':
            cnn_pretrained_model = models.swin_v2_t(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_v2_b':
            cnn_pretrained_model = models.swin_v2_b(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Swin_v2_s':
            cnn_pretrained_model = models.swin_v2_s(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vit':
            cnn_pretrained_model = models.vit_b_16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Wide_resnet':
            cnn_pretrained_model = models.wide_resnet101_2(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                    f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model


    #使用cnn进行特征提取，然后存储到adata.obsm["image_feat"]，再进行PCA降维后存储到adata.obsm["image_feat_pca"]
    def extract_image_feat(
        self,
        ):

        ##图像转换、增强部分
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],  #图像标准化
                          std =[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),#随机调整图像的对比度
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),#对图像进行高斯模糊
                          transforms.RandomInvert(),#随机反转图像的像素值
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)), #随机调整图像的清晰度
                          transforms.RandomSolarize(random.uniform(0, 1)), #随机将图像的像素值反转
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)), #随机进行仿射变换
                          transforms.RandomErasing()#随机擦除图像的一部分区域
                          ]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(mean=[0.54, 0.51, 0.68], 
        #                   std =[0.25, 0.21, 0.16])]

        # 将 transform_list 中的多个数据预处理操作按顺序组合在一起，形成一个串联的数据预处理管道.
        # 这样，可以将图像数据依次传递给该管道，然后依次经过 transform_list 中的操作进行处理。
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
        #model.fc = torch.nn.LeakyReLU(0.1)
        model.eval()

        # 通过image_crop（）函数可以得到adata.obs['slices_path']以及相对应的224*224大小的图像区域
        if "slices_path" not in self.adata.obs.keys():
             raise ValueError("Please run the function image_crop first")

        with tqdm(total=len(self.adata),
              desc="Extract image feature",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path) #打开文件
                spot_slice = spot_slice.resize((224,224)) #重构为224*224
                spot_slice = np.asarray(spot_slice, dtype="int32")  #转换为np.asarray
                spot_slice = spot_slice.astype(np.float32) #转换为float32
                tensor = img_to_tensor(spot_slice) #使用上述的img_to_tensor（）函数进行数据增强、转换
                tensor = tensor.resize_(1,3,224,224) #重构维度
                tensor = tensor.to(self.device)
                result = model(Variable(tensor)) #放到cnn中提取特征
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca clusterType of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata 

#用于剪裁图像，根据提供的坐标剪裁出50*50的图像区域，然后缩放至224*224大小的图像
def image_crop(
        adata,
        save_path,
        library_id=None,
        crop_size=50, #裁剪大小
        target_size=224, #目标大小
        verbose=False,
        ):
    if library_id is None:
       library_id = list(adata.uns["spatial"].keys())[0] #切片的序号

    #这段代码的作用是选择在 library_id 下，质量较高的组织切片对应的图像数据，
    # 并将它们存储在名为 image 的变量中。这个 image 变量是一个包含图像数据的数组，可以被用于后续的图像处理或分析。
    image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)# 将图像数据转换成 PIL（Python Imaging Library）图像
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            #设置图像范围
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            #img_pillow.crop（）用于对图像进行剪裁（裁剪）操作
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS) #用于将图像缩放到指定的大小，但保持其纵横比不变。利用三次插值Image.ANTIALIAS
            tile.resize((target_size, target_size)) #用于将 tile 图像缩放到目标大小 （50*50）
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata

def get_image_crop(
        ann_data,
        section_id,
        cnnType='ResNet50',
        pca_n_comps=50
):
    save_path_image_crop = Path(os.path.join('./', 'Image_crop', section_id))
    save_path_image_crop.mkdir(parents=True, exist_ok=True)
    ann_data = image_crop(ann_data, save_path=save_path_image_crop)
    ann_data = image_feature(ann_data, pca_components=pca_n_comps, cnnType=cnnType).extract_image_feat()
    return ann_data