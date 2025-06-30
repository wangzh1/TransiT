import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for NLoS transient complement', add_help=False)
    
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    
    parser.add_argument('--epochs', default=10, type=int)

    # Model params 
    parser.add_argument('--crop_hist', type=bool, )

    parser.add_argument('--crop_hist_start', default=100, type=int)
    parser.add_argument('--crop_hist_end', default=800, type=int)

    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--downsample_spatial', default=1, type=int)

    parser.add_argument('--mask_ratio', type=float, default=0.7,
                        help='mask ratio (default: 0.7)')

    parser.add_argument('--histogram_length', default=500, type=int)

    parser.add_argument('--histogram_num', default=1024, type=int)

    parser.add_argument('--loss', default='MSE')
    
    parser.add_argument('--embed_dim', default=512, type=int)

    parser.add_argument('--decoder_embed_dim', default=512, type=int)

    parser.add_argument('--decoder_num_heads', default=16, type=int)

    parser.add_argument('--num_heads', default=16, type=int)

    parser.add_argument('--depth', default=24, type=int)

    parser.add_argument('--decoder_depth', default=8, type=int)   
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
                        
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='Minimum learning raate')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--fc_drop_rate', type=float, default=0.0,
                        help='')
    # Dataset parameters
    parser.add_argument('--data_path', default='./', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    
    parser.add_argument('--seed', default=3407, type=int)
    
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument('--num_workers', default=0, type=int)
    
    parser.add_argument('--eval_freq', default=50, type=int)

    parser.add_argument('--save_freq', default=50, type=int)
    
    parser.add_argument('--log_metric', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    # Fintune training parameters
    parser.add_argument('--use_mean_pooling', default=False, type=bool,
                        help='If use mean pooling for classification')
    
    parser.add_argument('--jitter_std', type=float, default=6.0,
                        help='')
    parser.add_argument('--total_background_rate', type=float, default=0.01,
                        help='')

    return parser

