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
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def main(args):
    
    # fix the seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    accelerator = Accelerator(project_dir=args.output_dir)

    accelerator.print("{}".format(args).replace(', ', ',\n'))
    cudnn.benchmark = True
    ################################
    # Loading dataset
    realdataset = RealVideoMatDataDataset('./NLOSVideoDynamic/DemoN2', n=1)

    ################################
    if args.log_dir is not None and accelerator.is_local_main_process:
        os.makedirs(args.log_dir, exist_ok=True)

    data_loader_real = torch.utils.data.DataLoader(
        realdataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    ##########################################################################
    model = NLOSVideoFlash(dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads)

    ##########################################################################

    accelerator.print(f"Resume checkpoint from {args.resume}.")
    checkpoint = torch.load(args.resume)

    model.load_state_dict(checkpoint['model'])

    accelerator.print(f"Loaded checkpoint.")
        
    ##########################################################################
    model, data_loader_real = accelerator.prepare(model, data_loader_real)
    # return

    ##########################################################################
    accelerator.print(f"Start.")
    start_time = time.time()
    counter = 0
    with torch.no_grad():
        if accelerator.is_local_main_process:
            for real_batch in data_loader_real:
                time0 = time.time()
                # print(real_batch.shape)
                real_batch = real_batch[:,0,:,:,:]
                real_batch =torch.rot90(real_batch, k=-1, dims=(2, 3))
                real_batch =torch.rot90(real_batch, k=-1, dims=(2, 3))
                
                transients_pred_real, _ = model(real_batch, if_syn=False)
                print(time.time()-time0)
                for i in range(args.batch_size):
                    plt.figure()
                    # 使用 matplotlib 保存图像
                    plt.imshow(transients_pred_real[i,0,:,:].detach().cpu().numpy().squeeze(), cmap='inferno')  # 转换为 NumPy 数组并显示为灰度图
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig(os.path.join(args.log_dir, f'windmill_{counter}.png'), bbox_inches='tight', pad_inches=0)  # 保存图像
                    plt.show()  # 显示图像
                    plt.close()
                    counter = counter + 1
                            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Done time {}'.format(total_time_str))
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
