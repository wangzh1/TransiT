import os
import time
import torch
import datetime

import numpy as np
import torch.backends.cudnn as cudnn
from util.config_parser import get_args_parser
from util.lr_scheduler import cosine_scheduler
# import util.misc as misc
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from transients_dataset import *
from models_imaging import NLOSVideoFlash
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
from accelerate import Accelerator
from util.tflct import *
from util.tffk import *
from util.tfphasor import *
import tqdm
from sklearn.decomposition import PCA


def main(args):
    
    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    accelerator = Accelerator(project_dir=args.output_dir)
    ################################
    # Loading dataset
    dataset = NLOSVideoDatasetTest(root_dir=args.data_path, sequence_length=1)
    print(len(dataset))
    ################################
    if args.log_dir is not None and accelerator.is_local_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ##########################################################################
    model = NLOSVideoFlash(dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads).cuda()

    ##########################################################################

    checkpoint = torch.load(args.resume)
    
    model.load_state_dict(checkpoint['model'], strict=True)

    ##########################################################################
    model,data_loader = accelerator.prepare(model,data_loader)
    # return

    ##########################################################################
    accelerator.print(f"Start.")
    start_time = time.time()
    counter = 0
    with torch.no_grad():
        if accelerator.is_local_main_process:
            for (batch, images) in data_loader:
                time0 = time.time()
                transients_pred_real = model(batch)
                print(time.time()-time0)
                for i in range(args.batch_size):
                    plt.figure()
                    # 使用 matplotlib 保存图像
                    plt.imshow(transients_pred_real[i,0,:,:].detach().cpu().numpy().squeeze(), cmap='inferno')  # 转换为 NumPy 数组并显示为灰度图
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(os.path.join(args.log_dir, f'sim_{counter}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
                    plt.show()  # 显示图像
                    plt.close()
                    
                    plt.figure()
                    # 使用 matplotlib 保存图像
                    plt.imshow(images[i,0,:,:].detach().cpu().numpy().squeeze(), cmap='inferno')  # 转换为 NumPy 数组并显示为灰度图
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(os.path.join(args.log_dir, f'GT_{counter}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
                    plt.show()  # 显示图像
                    plt.close()                   
                    counter = counter + 1
                            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Done time {}'.format(total_time_str))

    return


    # if accelerator.is_local_main_process:
    #     for j in range(10, 11):
    #         with torch.no_grad():
    #             ################################
    #             for target_data,_  in data_loader_test:
    #                 # source_data 的形状为 (num_source_samples, feature_dim)
    #                 # print(target_data.shape)
    #                 _, target_features  = model(target_data)  # 源域特征
    #                 break  # 已获取所有源域数据


    #             # transients = rearrange(batch, 'b t n -> b n t').contiguous()
    #             # print(batch.shape)
    #             transients_pred, source_features  = model(batch.cuda() * j / 10)

    #             ################################
    #             # # 将数据从 PyTorch 张量转换为 NumPy 数组，以便使用 scikit-learn 的 PCA
    #             # source_features_np = source_features.cpu().numpy()
    #             # target_features_np = target_features.cpu().numpy()
                
    #             # source_features_np = source_features_np.squeeze()
    #             # target_features_np = target_features_np.reshape([-1, args.embed_dim])
    #             # # print(1,source_features_np.shape,target_features_np.shape)
    #             # # 定义 PCA 的子空间维度
    #             # subspace_dim = args.embed_dim  # 子空间维度，可根据需要调整

    #             # # 对源域特征进行 PCA
    #             # pca_source = PCA(n_components=subspace_dim)
    #             # pca_source.fit(source_features_np)
    #             # source_subspace = pca_source.components_.T  # 形状为 (feature_dim, subspace_dim)

    #             # # 对目标域特征进行 PCA
    #             # pca_target = PCA(n_components=subspace_dim)
    #             # pca_target.fit(target_features_np)
    #             # target_subspace = pca_target.components_.T  # 形状为 (feature_dim, subspace_dim)

    #             # # 计算子空间对齐矩阵 M
    #             # M = torch.from_numpy(source_subspace.T @ target_subspace).cuda()  # 形状为 (subspace_dim, subspace_dim)

    #             # # 将源域特征映射到目标域子空间
    #             # # 1. 将源域特征投影到源域子空间
    #             # source_features_subspace = source_features @ torch.from_numpy(source_subspace).cuda()  # (num_source_samples, subspace_dim)

    #             # # 2. 应用对齐矩阵 M
    #             # source_features_aligned = source_features_subspace @ M  # (num_source_samples, subspace_dim)

    #             # print(source_features_aligned.shape)
    #             # # return
    #             # ################################
    #             # transients_pred = model.module.Unzip(source_features_aligned)
    #             # # print(transients_pred.shape)
    #             # # return

    #             ################################

    #             plt.figure()
    #             # 使用 matplotlib 保存图像
    #             plt.imshow(transients_pred[0,0,:,:].detach().cpu().numpy().squeeze(), cmap='inferno')  # 转换为 NumPy 数组并显示为灰度图
    #             plt.axis('off')  # 关闭坐标轴
    #             plt.savefig(os.path.join(args.log_dir, f'Test_imaging_LCT_{j}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
    #             plt.show()  # 显示图像
    #             plt.close()

    #     return


    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
