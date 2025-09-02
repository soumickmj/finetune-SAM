#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
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
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss
from utils.dsc import dice_coeff
import cv2
import monai
from utils.utils import vis_image
import cfg

def main(args,test_image_list):
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list,phase='val',targets=['all'],if_prompt=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    cls_num = 2 # edit the class num 
    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla'
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=cls_num)
    elif args.finetune_type = 'lora':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=cls_num)
        sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(cls_num,dtype=torch.float)
    cls_dsc = torch.zeros(cls_num,dtype=torch.float)
    eps = 1e-9
    dsc_img = []
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []

    for i,data in enumerate(testloader,1):
        imgs = data['image'].to('cuda')
        msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
        msks = msks.to('cuda')
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= sam_fine_tune.image_encoder(imgs)

            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
            pred_fine, _ = sam_fine_tune.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )

        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())
        pred_fine = pred_fine[:,1,:,:]
        yhat = (pred_fine>=0).cpu().flatten()
        y = msks.cpu().flatten()

        for j in range(2):
            y_bi = y==j
            yhat_bi = yhat==j
            I = ((y_bi*yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi,yhat_bi).sum()).item()
            class_iou[j] += I/(U+eps)

        for cls in range(cls_num):
            mask_pred_cls = ((pred_fine>=0).cpu()==cls).float()
            mask_gt_cls = (msks.cpu()==cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls,mask_gt_cls).item()
        #print(i)

        dsc_img.append(dice_coeff(mask_pred_cls,mask_gt_cls).item())
    class_iou /=(i+1)
    cls_dsc /=(i+1)

    save_folder = os.path.join('test_results',args.dir_checkpoint)
    Path(save_folder).mkdir(parents=True,exist_ok = True)
    #np.save(os.path.join(save_folder,'test_masks.npy'),np.concatenate(pred_msk,axis=0))
    #np.save(os.path.join(save_folder,'test_name.npy'),np.concatenate(np.expand_dims(img_name_list,0),axis=0))


    print(dataset_name)      
    print('class dsc:',cls_dsc)      
    print('class iou:',class_iou)
if __name__ == "__main__":
    args = cfg.parse_args()
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    test_img_list = args.img_folder + dataset_name + '/val_5shot.csv'
    main(args,test_img_list)