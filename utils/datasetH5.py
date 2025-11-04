import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import pickle
import nibabel as nib
from scipy.ndimage import zoom
import einops
from utils.funcs import *
from torchvision.transforms import InterpolationMode
import SimpleITK as sitk
#from .utils.transforms import ResizeLongestSide
from utils.process_img import read_h5_data

def recursive_read_h5(group, path="", filter_names=None, satisfy_all_filts=True):
    ds_ls = []
    
    for key in group.keys():
        current_path = f"{path}/{key}" if path else key
        item = group[key]
        
        if isinstance(item, h5py.Dataset):
            # It's a dataset, check if it matches the filter(s)
            if filter_names is None:
                ds_ls.append(current_path)
            elif isinstance(filter_names, str):
                # Single filter name
                if filter_names in current_path:
                    ds_ls.append(current_path)
            elif isinstance(filter_names, list):
                # Multiple filter names
                if satisfy_all_filts:
                    # All filter names must be present
                    if all(filter_name in current_path for filter_name in filter_names):
                        ds_ls.append(current_path)
                else:
                    # Any filter name can be present
                    if any(filter_name in current_path for filter_name in filter_names):
                        ds_ls.append(current_path)
        elif isinstance(item, h5py.Group):
            # It's a group, recurse into it
            nested_datasets = recursive_read_h5(item, current_path, filter_names, satisfy_all_filts)
            ds_ls.extend(nested_datasets)

    return ds_ls

class Public_H5dataset(Dataset):
    def __init__(self,args, h5_path, phase='train',sample_num=50,channel_num=1,normalize_type='sam',crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=-1,if_prompt=True,prompt_type='point',region_type='largest_3',label_mapping=None,if_spatial=True,delete_empty_masks=True):
        '''
        target: 'combine_all': combine all the targets into binary segmentation
                'multi_all': keep all targets as multi-cls segmentation
                f'{one_target_name}': segmentation specific one type of target, such as 'hip'
        
        normalzie_type: 'sam' or 'medsam', if sam, using transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); if medsam, using [0,1] normalize
        cls: the target cls for segmentation
        prompt_type: point or box
        if_patial: if add spatial transformations or not
        
        '''
        super(Public_H5dataset, self).__init__()
        self.args = args
        self.h5_path = h5_path
        # self.img_folder = img_folder
        # self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.normalize_type = normalize_type
        self.targets = targets
        self.part_list = part_list
        self.cls = cls
        self.delete_empty_masks = delete_empty_masks
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.region_type = region_type
        self.label_dic = {}
        self.data_list = []
        self.label_mapping = label_mapping
        self.load_label_mapping()
        self.process_hd5()
        self.process_hdf5_label()
        self.if_spatial = if_spatial
        self.setup_transformations()

    def load_label_mapping(self):
        # Load the predefined label mappings from a pickle file
        # the format is {'label_name1':cls_idx1, 'label_name2':,cls_idx2}
        if self.label_mapping is None and bool(self.args.label_mapping):
            self.label_mapping = self.args.label_mapping
        if self.label_mapping:
            with open(self.label_mapping, 'rb') as handle:
                self.segment_names_to_labels = pickle.load(handle)

            if bool(self.targets):
                self.segment_names_to_labels = {name: idx for name, idx in self.segment_names_to_labels.items() if name in self.targets or name == 'background'}

            self.label_dic = {}
            for name, idx in self.segment_names_to_labels.items():
                if idx not in self.label_dic:
                    self.label_dic[idx] = []
                self.label_dic[idx].append(name)
            self.label_name_list = list(self.label_dic.keys())

            try:
                bg_name, bg_idx = next((name, idx) for name, idx in self.segment_names_to_labels.items() if 'background' in name.lower())
            except StopIteration:
                print("No explicit 'background' class found. Assigning index 0 as background.")
                if 0 in self.segment_names_to_labels.values():
                    raise ValueError(f"Ambiguous background: Index 0 is used by {self.label_dic.get(0)} but no class is named 'background'.")
                bg_name, bg_idx = 'background', 0
                self.segment_names_to_labels[bg_name] = bg_idx
                self.label_dic[bg_idx] = [bg_name]
            
            fg_indices = sorted(list(set(idx for name, idx in self.segment_names_to_labels.items() if idx != bg_idx)))
                            
            self.remapping_dict = {bg_idx: 0}
            for i, idx in enumerate(fg_indices):
                self.remapping_dict[idx] = i + 1
        
        else:
            self.segment_names_to_labels = {}
            self.remapping_dict = {i: i for i in range(self.args.num_cls)}
            self.label_dic = {}
        
        self.mask_remapper = np.vectorize(lambda x: self.remapping_dict.get(x, 0), otypes=[np.uint8])
        self.mask_unremapper = np.vectorize(lambda x: {v: k for k, v in self.remapping_dict.items()}.get(x, 0), otypes=[np.uint8])

    def process_hd5(self):
        with h5py.File(self.h5_path, 'r', swmr=True) as h5_file:

            self.data_list = recursive_read_h5(h5_file, filter_names=self.args.h5_filternames.split(','), satisfy_all_filts=self.args.h5filts_satisfy_all)
            print(f'Filtered data list to {len(self.data_list)} entries.')

    def process_hdf5_label(self):
        if os.path.exists(self.args.h5_path.replace('data.h5','meta_mask.h5')):
            self.mask_h5_path = self.args.h5_path.replace('data.h5','meta_mask.h5')
            with h5py.File(self.mask_h5_path, 'r', swmr=True) as h5_file:
                mask_list = recursive_read_h5(h5_file)
                print(f'{len(mask_list)} masks found.')
            #find common elements in self.data_list and mask_list
            common_list = list(set(self.data_list) & set(mask_list))
            print(f'{len(common_list)} common elements found between data and masks, keeping only them!')
            self.data_list = common_list
        else:
            self.mask_h5_path = None

    def should_keep(self, msk, mask_path):
        """
        Determine whether to keep an image based on the mask and part list conditions.
        """
        mask_array = np.array(msk, dtype=int)
        #print(np.unique(mask_array))
        if 'combine_all' in self.targets:
            return np.any(mask_array > 0)
        elif 'multi_all' in self.targets:
            return np.any(mask_array > 0)
        elif any(target in self.targets for target in self.segment_names_to_labels):
            # target_classes = [self.segment_names_to_labels[target][1] for target in self.targets if target in self.segment_names_to_labels] #Original line: Not sure why they extracted the 2nd element from the mapping, because they anyway mention the format as {'label_name1':cls_idx1, 'label_name2':,cls_idx2} and we need the cls_idxX
            target_classes = [self.segment_names_to_labels[target] for target in self.targets if target in self.segment_names_to_labels]
            return any(np.any(mask_array == cls) for cls in target_classes)
        elif self.cls>0:
            return np.any(mask_array == self.cls)
        if self.part_list[0] != 'all':
            return any(part in mask_path for part in self.part_list)
        return False

    def setup_transformations(self):
        if self.phase =='train':
            transformations = [transforms.RandomEqualize(p=0.1),
                 transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                              ]
            # if add spatial transform 
            if self.if_spatial:
                self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1.5), interpolation=InterpolationMode.NEAREST),
                         transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])
        else:
            transformations = []
        transformations.append(transforms.ToTensor())
        if self.normalize_type == 'sam':
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        elif self.normalize_type == 'medsam':
            transformations.append(transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))))
        self.transform_img = transforms.Compose(transformations)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        tag = self.data_list[index]
        img_msk = read_h5_data(h5_path=self.h5_path, dataset_path=tag, pth_lbl=self.mask_h5_path, slice_index=self.args.slice_index, norm_type=self.args.prenorm_type, window_min_percentile=self.args.prenorm_window_min_percentile, window_max_percentile=self.args.prenorm_window_max_percentile)
        if len(img_msk)==3:
            img, msk, prepad_shape = img_msk
        else:
            img, prepad_shape = img_msk
            msk = np.zeros_like(img, dtype=np.uint8) #no mask provided, create a dummy/placeholder mask
        
        img = Image.fromarray(img).convert('RGB')

        if  np.any(msk) and (not ('combine_all' in self.targets or 'multi_all' in self.targets)):
            msk = self.mask_remapper(msk)
        msk = Image.fromarray(msk).convert('L')

        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        img, msk = self.apply_transformations(img, msk)

        if 'combine_all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        elif 'multi_all' in self.targets:
            msk = np.array(msk,dtype=int)
        elif self.cls>0:
            msk = np.array(msk==self.cls,dtype=int)

        return self.prepare_output(img, msk, tag, prepad_shape=prepad_shape)
    
    def apply_transformations(self, img, msk):
        if self.crop:
            img, msk = self.apply_crop(img, msk)
        img = self.transform_img(img)
        msk = torch.tensor(np.array(msk, dtype=int), dtype=torch.long)

        if self.phase=='train' and self.if_spatial:
            mask_cls = np.array(msk,dtype=int)
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)
            msk = torch.tensor(mask_cls)
        return img, msk

    def apply_crop(self, img, msk):
        t, l, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk

    def prepare_output(self, img, msk, img_path, coords=None, prepad_shape=None):
        if len(msk.shape)==2:
            # msk = torch.unsqueeze(torch.tensor(msk,dtype=torch.long),0)
            msk = torch.unsqueeze(msk.clone(), 0).long() #due to UserWarning
        output = {'image': img, 'mask': msk, 'ds_path': img_path, 'prepad_shape': prepad_shape}
        if self.if_prompt:
            # Assuming get_first_prompt and get_top_boxes functions are defined and handle prompt creation
            if self.prompt_type == 'point':
                prompt, mask_now = get_first_prompt(msk.numpy()[0], dist_thre_ratio=self.args.prompt_dist_thre_ratio, region_type=self.region_type)
                pc = torch.tensor(prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                output.update({'point_coords': pc, 'point_labels': pl,'mask':msk})
            elif self.prompt_type == 'box':
                if coords is not None:
                    prompt = np.array([[int(x) for x in group.split('-')] for group in coords.split(';')])
                    bbox_mode = "supplied"
                else:
                    prompt, mask_now = get_top_boxes(msk.numpy()[0], dist_thre_ratio=self.args.prompt_dist_thre_ratio, region_type=self.region_type)
                    msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                    bbox_mode = "computed"
                box = torch.tensor(prompt, dtype=torch.float)
                output.update({'boxes': box,'mask':msk, 'bbox_mode': bbox_mode})
            elif self.prompt_type == 'hybrid':
                point_prompt, _ = get_first_prompt(msk[0].numpy(), dist_thre_ratio=self.args.prompt_dist_thre_ratio, region_type=self.region_type)
                box_prompt, _ = get_top_boxes(msk.numpy(), dist_thre_ratio=self.args.prompt_dist_thre_ratio, region_type=self.region_type)
                pc = torch.tensor(point_prompt[:, :2], dtype=torch.float)
                pl = torch.tensor(point_prompt[:, -1], dtype=torch.float)
                box = torch.tensor(box_prompt, dtype=torch.float)
                output.update({'point_coords': pc, 'point_labels': pl, 'boxes': box})
        return output
