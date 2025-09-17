#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
import pandas as pd
import h5py
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss, iou_multiclass
from utils.dsc import dice_coeff
import cv2
import monai
from utils.utils import vis_image
import cfg
from argparse import Namespace
import json
from utils.process_img import unpad_arr, save_image, post_process_mask, create_overlay

cfg.set_seed(1701)  # Set a fixed seed for reproducibility

def main(args,test_image_list):
    # change to 'combine_all' if you want to combine all targets into 1 cls
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_image_list,phase='test',targets=args.targets,if_prompt=False,delete_empty_masks=False) #for testing, we will consider all the samples (including empty masks)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    if args.test_prefinetune:
        print('Testing pre-finetuned model..')
        if args.test_tag:
            args.test_tag = f'{args.test_tag}_prefinetune'
        else:
            args.test_tag = 'prefinetune'
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
    else:
        if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
            sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
        elif args.finetune_type == 'lora':
            sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
            sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
            sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(args.num_cls,dtype=torch.float)
    cls_dsc = torch.zeros(args.num_cls,dtype=torch.float)
    eps = 1e-9
    img_name_list = []
    pred_msk = []

    if bool(args.seg_save_dir):        
        os.makedirs(args.seg_save_dir, exist_ok=True)

    emb_storage = []
    iou_storage = []

    for i,data in enumerate(tqdm(testloader)):
        imgs = data['image'].to('cuda')
        msks = torchvision.transforms.Resize((args.out_size,args.out_size),InterpolationMode.NEAREST)(data['mask'])
        msks = msks.to('cuda')
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= sam_fine_tune.image_encoder(imgs)

            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
            pred_fine, iou_predictions = sam_fine_tune.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            
        if args.store_emb:
            emb_storage.append({
                "image_name": data['img_name'][0],
                "img_emb": img_emb.detach().cpu().numpy(),
                "sparse_emb": sparse_emb.detach().cpu().numpy(),
                "dense_emb": dense_emb.detach().cpu().numpy()
            })

        if pred_fine.shape[-1] != args.out_size or pred_fine.shape[-2] != args.out_size: #SAM's output is always 256x256, resize it to the original size
            pred_fine = F.interpolate(pred_fine, size=(args.out_size, args.out_size), mode='bilinear')

        true_iou = iou_multiclass(pred_fine, msks, args.num_cls, eps=1e-7)
        iou_storage.append((data['img_name'][0], iou_predictions.cpu().numpy(), true_iou.cpu().numpy()))

        pred_fine = pred_fine.argmax(dim=1)

        if list(pred_fine.shape[-2:]) != [s.item() for s in data['prepad_shape']]:
            pred_fine = unpad_arr(pred_fine, [s.item() for s in data['prepad_shape']]) 
            msks = unpad_arr(msks, [s.item() for s in data['prepad_shape']])

        if bool(args.seg_save_dir):
            pred_mask = np.squeeze(pred_fine.cpu().numpy())
            pred_mask = test_dataset.mask_unremapper(pred_mask) #if we have re-mapped the IDs inside the dataset, we need to invert the operation now
            save_image(pred_mask, os.path.join(args.seg_save_dir, data['img_name'][0]), is_RGB=False)
            if args.post_process_mask:
                proc_mask = post_process_mask(pred_mask, fill_holes=args.post_process_fillholes, keep_largest_component=args.post_process_largestsegment)
                save_image(proc_mask, os.path.join(args.seg_save_dir, args.proc_tag, data['img_name'][0]), is_RGB=False)


        pred_msk.append(pred_fine.cpu().numpy())
        yhat = (pred_fine).cpu().long().flatten()
        y = msks.cpu().flatten()

        for j in range(args.num_cls):
            y_bi = y==j
            yhat_bi = yhat==j
            I = ((y_bi*yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi,yhat_bi).sum()).item()
            class_iou[j] += I/(U+eps)

        for cls in range(args.num_cls):
            mask_pred_cls = ((pred_fine).cpu()==cls).float()
            mask_gt_cls = (msks.cpu()==cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls,mask_gt_cls).item()
        #print(i)

    class_iou /=(i+1)
    cls_dsc /=(i+1)

    save_subfolder = 'test_results'
    if args.test_tag:
        save_subfolder += f'_{args.test_tag}'

    save_folder = os.path.join(args.dir_checkpoint, save_subfolder)
    Path(save_folder).mkdir(parents=True,exist_ok = True)
    np.save(os.path.join(save_folder,'test_masks.npy'),np.concatenate(pred_msk,axis=0))
    np.save(os.path.join(save_folder,'test_name.npy'),np.concatenate(np.expand_dims(img_name_list,0),axis=0))

    inverse_remapping = {v: k for k, v in test_dataset.remapping_dict.items()}
    labels_to_names = {v: k for k, v in test_dataset.segment_names_to_labels.items()}
    num_iou_cols = iou_storage[0][1].shape[1] # Gets the number of columns in the results
    iou_column_names = ["predIOU_"+labels_to_names[inverse_remapping[i]] for i in range(num_iou_cols)]
    trueiou_column_names = ["trueIOU_"+labels_to_names[inverse_remapping[i]] for i in range(num_iou_cols)]
    data_for_df = [
        [filename] + list(iou_array.flatten()) + list(trueiou_array.flatten())
        for filename, iou_array, trueiou_array in iou_storage
    ]
    all_column_names = ['filename'] + iou_column_names + trueiou_column_names
    iou_df = pd.DataFrame(data_for_df, columns=all_column_names)
    iou_df.to_csv(os.path.join(save_folder,'test_SAM_noRefIOUs.csv'), index=False)

    if args.store_emb:
        with h5py.File(os.path.join(save_folder,'embs.h5'), 'w') as hf:
            for item in emb_storage:
                group = hf.create_group(item["image_name"])
                group.create_dataset("img_emb", data=item["img_emb"])
                group.create_dataset("sparse_emb", data=item["sparse_emb"])
                group.create_dataset("dense_emb", data=item["dense_emb"])

    print(dataset_name)      
    print('class dsc:',cls_dsc)      
    print('class iou:',class_iou)

if __name__ == "__main__":
    args = cfg.prepare_args(cfg.parse_args())
    cfg.set_seed(args.seed)  # Set a fixed seed for reproducibility (again, to use the supplied seed)

    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name))
    
    args_path = f"{args.dir_checkpoint}/args.json"

    # Reading the args from the json file
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    
    # Converting dictionary to Namespace
    args_orig = Namespace(**args_dict)        
    args_orig.test_img_list = args.test_img_list
    args_orig.seg_save_dir = args.seg_save_dir
    args_orig.test_prefinetune = args.test_prefinetune
    args_orig.test_tag = args.test_tag
    args_orig.store_emb = args.store_emb
    args_orig.post_process_mask = args.post_process_mask
    args_orig.post_process_fillholes = args.post_process_fillholes
    args_orig.post_process_largestsegment = args.post_process_largestsegment

    if args.post_process_mask:
        args_orig.proc_tag = 'proc'
        if args.post_process_fillholes:
            args_orig.proc_tag += '_fillholes'
        if args.post_process_largestsegment:
            args_orig.proc_tag += '_largestseg'
        os.makedirs(os.path.join(args_orig.seg_save_dir, args_orig.proc_tag), exist_ok=True)
    
    if args_orig.seed != args.seed:
        print(f"Warning: The seed in the config file ({args_orig.seed}) does not match the one provided ({args.seed}). Using the provided seed.")

    main(args_orig, args_orig.test_img_list)