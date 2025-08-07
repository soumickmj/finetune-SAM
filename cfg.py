import argparse
import os
import random
import numpy as np
import torch

label_mapping_ukbabd = {
    'liver': 1,
    'gallbladder': 2,
    'kidney': 3,
    'pancreas': 4,
    'aorta': 5,
    'spleen': 6
}

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1701, help='random seed for reproducibility')

    parser.add_argument('--net', type=str, default='sam', help='net type')
    parser.add_argument('--arch', type=str, default='vit_b', help='net architecture, pick between vit_h, vit_b, vit_t')
    parser.add_argument('--baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('--dataset_name', type=str, default='MRI-Prostate', help='the name of dataset to be finetuned')
    
    parser.add_argument('--img_folder', type=str, default='', help='the folder putting images')
    parser.add_argument('--mask_folder', type=str, default='', help='the folder putting masks')
    parser.add_argument('--train_img_list', type=str, default='')
    parser.add_argument('--val_img_list', type=str,default='')
    parser.add_argument('--targets', type=str,default='liver')

    parser.add_argument('--loss_mode', type=int, default=0, help='0: original (Dice + CE), 1: Dice + Focal, 2: Tversky + Focal')
    parser.add_argument('--include_background_loss', action=argparse.BooleanOptionalAction, default=True, help='Default:True. If False, channel index 0 (background category) is excluded from the calculation. if the non-background segmentations are small compared to the total image size they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence.')
    parser.add_argument('--add_boundary_loss', action=argparse.BooleanOptionalAction, default=False, help='additionally apply boundary loss, Dice computed ')
    parser.add_argument('--val_dsc_monitor', action=argparse.BooleanOptionalAction, default=True, help='whether to monitor the validation DSC (default) or validation loss')

    parser.add_argument('--finetune_type', type=str, default='adapter', help='normalization type, pick among vanilla,adapter,lora')
    parser.add_argument('--normalize_type', type=str, default='sam', help='normalization type, pick between sam or medsam')
    
    parser.add_argument('--dir_checkpoint', type=str, default='checkpoints', help='the checkpoint folder to save final model')
    parser.add_argument('--num_cls', type=int, default=2, help='the number of output channels (need to be your target cls num +1)')
    parser.add_argument('--epochs', type=int, default=200, help='the number of largest epochs to train')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b_01ec64.pth', help='the path to the checkpoint to load')
    
    parser.add_argument('--type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('--vis', type=int, default=None, help='visualization')
    parser.add_argument('--reverse', action=argparse.BooleanOptionalAction, default=False, help='adversary reverse')
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, default=False, help='adversary reverse')
    parser.add_argument('--val_freq',type=int,default=100,help='interval between each validation')
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=True, help='use gpu or not')
    parser.add_argument('--gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('--sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('--epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('--image_size', type=int, default=1024, help='image_size')
    parser.add_argument('--out_size', type=int, default=256, help='output_size')
    parser.add_argument('--patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('--dim', type=int, default=512, help='dim_size')
    parser.add_argument('--depth', type=int, default=64, help='depth')
    parser.add_argument('--heads', type=int, default=16, help='heads number')
    parser.add_argument('--mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('--w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('--s', action=argparse.BooleanOptionalAction, default=True, help='whether shuffle the dataset')
    parser.add_argument('--if_warmup', action=argparse.BooleanOptionalAction, default=True, help='if warm up training phase')
    parser.add_argument('--warmup_period', type=int, default=200, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('--imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('--weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('--base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('--sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('--distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('--dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('--thd', action=argparse.BooleanOptionalAction, default=False , help='3d or not')
    parser.add_argument('--chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('--num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('--roi_size', type=int, default=96 , help='resolution of roi')

    parser.add_argument('--if_update_encoder', action=argparse.BooleanOptionalAction, default=True , help='if update_image_encoder')
    parser.add_argument('--if_encoder_adapter', action=argparse.BooleanOptionalAction, default=False , help='if add adapter to encoder')
    
    parser.add_argument('--encoder_adapter_depths', type=list, default=list(range(0,12)) , help='the depth of blocks to add adapter')
    parser.add_argument('--if_mask_decoder_adapter', action=argparse.BooleanOptionalAction, default=False , help='if add adapter to mask decoder')
    parser.add_argument('--decoder_adapt_depth', type=int, default=2, help='the depth of the decoder adapter')
    
    parser.add_argument('--if_encoder_lora_layer', action=argparse.BooleanOptionalAction, default=False , help='if add lora to encoder')
    parser.add_argument('--if_decoder_lora_layer', action=argparse.BooleanOptionalAction, default=False , help='if add lora to decoder')
    parser.add_argument('--encoder_lora_layer', type=list, default=[] , help='the depth of blocks to add lora, if [], it will add at each layer')
    
    parser.add_argument('--if_split_encoder_gpus', action=argparse.BooleanOptionalAction, default=False , help='if split encoder to multiple gpus')
    parser.add_argument('--devices', type=list, default=[0,1] , help='if split encoder to multiple gpus')
    parser.add_argument('--gpu_fractions', type=list, default=[0.5,0.5] , help='how to split encoder to multiple gpus')
    
  
    parser.add_argument('--evl_chunk', type=int, default=None , help='evaluation chunk')
    
    #added by Soumick
    parser.add_argument('--run_tag', type=str, default="" , help='tag to append to the run name')
    parser.add_argument('--load_all', action=argparse.BooleanOptionalAction, default=True, help='No prechecks of the masks will be performed, all masks will be loaded')
    parser.add_argument('--slice_index', type=int, default=-1, help='slice/channel (last dim) index for 3D images')
    parser.add_argument('--label_mapping', default="/group/glastonbury/soumick/dataset/ukbbnii/minisets/classlabel_mapping.pkl" , help='Path to the label mapping file. Leave empty for no mapping.')
    parser.add_argument('--prenorm_type', type=str, default=None, help='pre-normalisation type for input images, pick between minmax, window. To do no pre-normalisation, set to None.')
    parser.add_argument('--prenorm_window_min_percentile', type=int, default=1, help='min percentile for window normalization')
    parser.add_argument('--prenorm_window_max_percentile', type=int, default=99, help='max percentile for window normalization')
    parser.add_argument('--test_img_list', type=str,default='', help='the list of test images, to be used only by val_finetune_noprompt.py')
    parser.add_argument('--test_prefinetune', action=argparse.BooleanOptionalAction, default=False, help='whether to use pre-finetuning weights (i.e. sam_ckpt) for testing')
    parser.add_argument('--test_tag', type=str, default='', help='tag to append to the test_results folder name')
    parser.add_argument('--seg_save_dir', type=str, default='', help='directory to save segmentation results')
    parser.add_argument('--prompt_region_type', type=str, default='', help='type of prompt region. Options: random, all, largest_k. If empty, no prompts will be generated.')
    parser.add_argument('--no_bbox_input', action=argparse.BooleanOptionalAction, default=False, help='while using bbox-mode (from the perspective of the DS), whether to supply bounding box prompts to the model')
    parser.add_argument('--prompt_dist_thre_ratio', type=float, default=0.1, help='ratio of distance threshold for prompt generation (for bbox, it is the the randomness at each side of box). 0 means always a perfect prompt')

    opt = parser.parse_args()

    return opt

def prepare_args(args):
    args.dir_checkpoint = os.path.join(args.dir_checkpoint, args.dataset_name + "_" + args.run_tag)
    if args.targets:
        args.targets = args.targets.split(',')
        assert args.num_cls == len(args.targets) + 1, "num_cls should be equal to the number of targets + 1 (for background)"
    return args

def set_seed(seed=1701):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # NumPy
    np.random.seed(seed)
    
    # Python random
    random.seed(seed)
    
    # OS environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)