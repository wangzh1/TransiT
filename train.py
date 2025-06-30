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

def compute_mmd_loss(x_src, x_tgt, kernel='rbf', sigma=1.0):

    if kernel == 'linear':
        K_xx = torch.matmul(x_src, x_src.t())
        K_yy = torch.matmul(x_tgt, x_tgt.t())
        K_xy = torch.matmul(x_src, x_tgt.t())
    elif kernel == 'rbf':
        def gaussian_kernel(a, b, sigma):
            dist = torch.cdist(a, b, p=2) ** 2
            return torch.exp(-dist / (2 * sigma ** 2))
        K_xx = gaussian_kernel(x_src, x_src, sigma)
        K_yy = gaussian_kernel(x_tgt, x_tgt, sigma)
        K_xy = gaussian_kernel(x_src, x_tgt, sigma)
    else:
        raise ValueError('Unsupported kernel type')

    m = x_src.size(0)
    n = x_tgt.size(0)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def coral_loss(source, target):

    d = source.size(1)
    # 计算源域和目标域的协方差矩阵
    xm = source - source.mean(0, keepdim=True)
    xc = (xm.t() @ xm) / (source.size(0) - 1)

    xmt = target - target.mean(0, keepdim=True)
    xct = (xmt.t() @ xmt) / (target.size(0) - 1)

    # 计算协方差矩阵差异的弗罗贝尼乌斯范数
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss


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
    dataset = NLOSVideoDatasetTest(root_dir=args.data_path, sequence_length=2)
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    realdataset = RealVideoMatDataDataset('./NLOSVideoDynamic/Man', n=2)
    accelerator.print("Sampler_train = %s" % str(train_size))

    ################################
    if args.log_dir is not None and accelerator.is_local_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_real = torch.utils.data.DataLoader(
        realdataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    ##########################################################################
    model = NLOSVideoFlash(dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    criterion = torch.nn.MSELoss()
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_training_steps_per_epoch = len(train_dataset) // args.batch_size
    accelerator.print('number of params: {} M'.format(n_parameters / 1e6))
    accelerator.print("LR = %.8f" % args.lr)
    accelerator.print("Batch size = %d" % args.batch_size)
    accelerator.print("Number of training steps = %d" % num_training_steps_per_epoch)
    accelerator.print("Number of training examples per epoch = %d" % (args.batch_size * num_training_steps_per_epoch))

    accelerator.print("Use step level LR & WD scheduler!")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*num_training_steps_per_epoch // 3 // 2, eta_min=args.min_lr)
        # Warmup scheduler
    def warmup_lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    # accelerator.print("Max LR = %.7f, Min LR = %.7f" % (max(lr_schedule_values), min(lr_schedule_values)))
    accelerator.print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    # return

    ##########################################################################
    if args.resume:
        accelerator.print(f"Resume checkpoint from {args.resume}.")
        checkpoint = torch.load(args.resume)
        
        model.load_state_dict(checkpoint['model'])
        
        accelerator.print(f"Loaded checkpoint.")
        
    ##########################################################################
    model, optimizer, data_loader_train, data_loader_test, data_loader_real = accelerator.prepare(model, optimizer, data_loader_train, data_loader_test, data_loader_real)
    # return

    ##########################################################################
    accelerator.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_loss =1e9
    total_step = 0

    # 初始化存储指标的列表
    train_losses = []
    eval_losses = []
    
    # LCT = lct(spatial= 64, crop=1024, bin_len=0.003, wall_size=1)
    # LCT.todev(dev=model.device,dnum=1)

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = 0
        local_step = 0
        
        real_data_iter = itertools.cycle(data_loader_real)
        
        pbar = tqdm.tqdm(
            enumerate(zip(data_loader_train, real_data_iter)),
            total=len(data_loader_train),
            disable=(not accelerator.is_main_process)
        )

        for i, ((train_batch, train_images), real_batch) in pbar:
            for j, param_group in enumerate(optimizer.param_groups):
                if total_step < args.warmup_steps:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
                    
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[total_step]
                if accelerator.is_local_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_writer.add_scalar('Learning Rate', current_lr, total_step)

            # with torch.no_grad():
            #     time0 = time.time()
            transients_pred, features = model(x_current = train_batch[:,1,:,:,:].squeeze(), x_previous = train_batch[:,0,:,:,:].squeeze(), if_syn=True)
            transients_pred_real, features_real = model(x_current = torch.rot90(real_batch[:,1,:,:,:].squeeze(), k=-1, dims=(2, 3)), x_previous = real_batch[:,0,:,:,:].squeeze(),if_syn=False)
            

            loss = criterion(transients_pred.squeeze(), train_images[:,1,:,:])
            # return
            optimizer.zero_grad()   
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item() 
            local_step += 1
            total_step += 1
            
            pbar.set_description(f"epoch {epoch + 1} iter {i}: train loss {loss.item():.6f}. lr {optimizer.param_groups[0]['lr']:e}")
            if accelerator.is_local_main_process:
                log_writer.add_scalar('Training loss', loss.item(), epoch * len(data_loader_train) + i)
                train_losses.append(loss.item())
                
        accelerator.print(f'Train epoch {epoch} done. Loss: {total_loss / num_training_steps_per_epoch}')
        
        ##########################################################################
        if accelerator.is_local_main_process:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
            for i in range(6):
                row = i // 3
                col = i % 3
                axes[row, col].imshow(train_images[i, 0, :, :].cpu().numpy().squeeze(), cmap='inferno')
                axes[row, col].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.log_dir, f'epoch_{epoch}_images_grid.png'), bbox_inches='tight', pad_inches=0) 
            # plt.show()
            plt.close()

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
            for i in range(6):
                row = i // 3
                col = i % 3
                axes[row, col].imshow(transients_pred[i, 0, :, :].detach().cpu().numpy().squeeze(), cmap='inferno')
                axes[row, col].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.log_dir, f'epoch_{epoch}_transients_pred_grid.png'), bbox_inches='tight', pad_inches=0)  
            # plt.show()
            plt.close()

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
            for i in range(6):
                row = i // 3
                col = i % 3
                axes[row, col].imshow(transients_pred_real[i, 0, :, :].detach().cpu().numpy().squeeze(), cmap='inferno')
                axes[row, col].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.log_dir, f'epoch_{epoch}_real_pred_grid.png'), bbox_inches='tight', pad_inches=0) 
            # plt.show()
            plt.close()
                            
        ##########################################################################
        accelerator.wait_for_everyone()

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs) and accelerator.is_main_process:

            unwrap_model = accelerator.unwrap_model(model)
            unwrap_optim = accelerator.unwrap_model(optimizer)
            # unwrap_lr = accelerator.unwrap_model(scheduler)
            torch.save({
            'model' : unwrap_model.state_dict(),
            'optimizer' : unwrap_optim.state_dict(),
            # 'lr_state' : unwrap_lr.state_dict()},
            'epoch': epoch,
            'args': args},
            args.output_dir + f'/ckpt_{epoch+1}.pt')
            accelerator.print(f'checkpoint ckpt_{epoch+1}.pt is saved...')

        if accelerator.is_main_process:
            # 绘制训练损失图
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.legend()
            plt.savefig(os.path.join(args.log_dir, f'visualizations_train_loss_epoch.png'))
            plt.close()

            plt.figure()
            plt.plot(eval_losses, label='Eval Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Eval Loss Over Epochs')
            plt.legend()
            plt.savefig(os.path.join(args.log_dir, f'visualizations_eval_loss_epoch.png'))
            plt.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Training time {}'.format(total_time_str))
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
