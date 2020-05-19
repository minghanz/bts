import argparse
import sys

from c3d.utils_general.argparse_f import init_argparser_f

def parse_args_main():
    parser = init_argparser_f(description='BTS Pytorch train.')

    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                        'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                                default='densenet161_bts')
    # Dataset
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--data_source',               type=str,   help='depth source, depth image (kitti_depth) or raw data (kitti_raw)', default='kitti_depth') # add by Minghan
    parser.add_argument('--init_width',                type=float, help='rescale the width to what at the beginning after kb cropping', default=0)
    parser.add_argument('--init_height',               type=float, help='rescale the height to what at the beginning after kb cropping', default=0)

    # Log and save
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)
    parser.add_argument('--log_freq_ini',              type=int,   help='Logging frequency in global steps in first 1000 iters', default=100)
    parser.add_argument('--print_freq',                type=int,   help='Printing frequency in global steps', default=100)
    parser.add_argument('--eval_time',                             help='if set, print timing of each iteration', action='store_true')
    parser.add_argument('--gpu_sync',                              help='if set, cuda.synchronization is used, for showing better timing', action='store_true')

    # Training
    parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
    parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
    parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
    parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--c3d_weight',                type=float, help='weight for continuous 3D loss in back proped in training') # add by Minghan
    parser.add_argument('--silog_weight',              type=float, help='weight for si_log loss in back proped in training') # add by Minghan
    parser.add_argument('--pho_weight',                type=float, help='weight for photometric loss in back proped in training') # add by Minghan
    parser.add_argument("--seq_frame_n_c3d",           type=int,   help="number of sequential frames for c3d loss (these frames are in the mini-batch)", default=1 )
    parser.add_argument("--seq_frame_n_pho",           type=int,   help="number of sequential frames for pho loss (these frames aren't in the mini-batch)", default=1 )
    parser.add_argument("--batch_same_intr",                       help="whether a mini-batch should come from the same day (with the same intrinsics)", action='store_true' )
    parser.add_argument("--turn_off_dloss",            type=int,   help="turn off depth loss after how many epochs to alleviate problem at edges, -1 to never disable depth loss", default=-1 )
    # parser.add_argument("--seq_aside",                             help="whether the sequential frames are in mini-batch or aside from training data (used only for image reconstruction)", action='store_true' )
    parser.add_argument("--other_scale",               type=int,   help="scaling down original image to make photometric error easier", default=-1 )
    parser.add_argument("--keep_velo",                             help="whether velodyne points should be kept in batch", action='store_true' )
    parser.add_argument("--side_full_img",                         help="if true, the side images will not be cropped for better image wrapping", action='store_true' )
    parser.add_argument("--depth_weight_decay",        type=float, help="the coefficient of the weight for depth loss decay every epoch", default=1 )
    parser.add_argument("--use_l1_loss",                           help="if true, use l1 loss instead of silog loss for depth", action='store_true' )
    parser.add_argument("--inbalance_to_closer",       type=float, help="if true, and use_l1_loss is true, give higher weight for over-estimated depth, to prefer closer prediction", default=1 )
    

    # Preprocessing
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    # Multi-gpu training
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
    # Online eval
    parser.add_argument('--eval_before_train',                     help='if set, perform online eval before the first training epoch', action='store_true')
    parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    # parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                        'if empty outputs to checkpoint folder', default='')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        # args = parser.parse_args([arg_filename_with_prefix])
        args, args_rest = parser.parse_known_args([arg_filename_with_prefix])
        return args, args_rest
    else:
        args = parser.parse_args()
        return args