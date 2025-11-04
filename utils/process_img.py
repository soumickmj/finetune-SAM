import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label as skLabel
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy.ndimage import label, binary_fill_holes
import matplotlib
import h5py

def get_images(pth_img, pth_lbl='', slice_index=None, norm_type=None, window_min_percentile=1, window_max_percentile=99):
    file_sitk = sitk.ReadImage(pth_img)
    image_data = sitk.GetArrayFromImage(file_sitk) #channel (or slice in 3D cases) first, then x, y. TODO: need to check when the data is more than 3D, e.g. 4D or 5D.

    if bool(pth_lbl):
        file_sitk_lbl = sitk.ReadImage(pth_lbl)
        label_data = np.uint8(sitk.GetArrayFromImage(file_sitk_lbl)) #same shape as image_data, but with labels instead of intensities
        assert image_data.shape == label_data.shape, "Image and label data must have the same shape. Current shapes: image_data={}, label_data={}".format(image_data.shape, label_data.shape)

    assert len(image_data.shape) in [2, 3], "Image data must be 2D or 3D, as the function get_image isn't implemented for more. Current shape: {}".format(image_data.shape)
    
    if slice_index not in [None, -1]:
        image_data = image_data[..., slice_index, :, :].squeeze()
        if bool(pth_lbl):
            label_data = label_data[..., slice_index, :, :].squeeze()

    # Check if the last 2 dimensions are not identical and pad to make it a perfect square
    height, width = image_data.shape[-2], image_data.shape[-1]
    if height != width:
        # Calculate padding needed to make it square
        max_dim = max(height, width)
        pad_height = max_dim - height
        pad_width = max_dim - width
        
        # Apply padding to the last two dimensions
        if len(image_data.shape) == 2:
            # 2D image
            image_data = np.pad(image_data, 
                                ((pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)
            if bool(pth_lbl):
                label_data = np.pad(label_data, 
                                    ((pad_height//2, pad_height - pad_height//2), 
                                    (pad_width//2, pad_width - pad_width//2)), 
                                    mode='constant', constant_values=0)
        elif len(image_data.shape) == 3:
            # 3D image (multiple slices)
            image_data = np.pad(image_data, 
                                ((0, 0), 
                                (pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)
            if bool(pth_lbl):
                label_data = np.pad(label_data, 
                                    ((0, 0), 
                                    (pad_height//2, pad_height - pad_height//2), 
                                    (pad_width//2, pad_width - pad_width//2)), 
                                    mode='constant', constant_values=0)

    # normalise to [0, 255]    
    if norm_type is not None:
        if norm_type == "minmax": # Method 0: Simple normalisation with Min/Max scaling
            image_data_pre = np.uint8((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255.0)
        elif norm_type == "window":  # Method 1: Window/Level adjustment (like CT/MRI viewing)
            lower_bound = np.percentile(image_data[image_data > 0], window_min_percentile) if np.any(image_data > 0) else 0
            upper_bound = np.percentile(image_data, window_max_percentile)
            image_data_windowed = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = np.uint8((image_data_windowed - lower_bound) / (upper_bound - lower_bound) * 255.0)
        else:
            raise ValueError("Unsupported normalisation type: {}".format(norm_type))
    else:
        image_data_pre = np.uint8(image_data * 255.0)

    if bool(pth_lbl):
        return image_data_pre, label_data, (height, width)
    else:
        return image_data_pre, (height, width)

def read_h5_data(h5_path, dataset_path, pth_lbl=None, slice_index=None, norm_type=None, window_min_percentile=1, window_max_percentile=99):
    with h5py.File(h5_path, 'r', swmr=True) as h5_file:
        image_data = h5_file[dataset_path][:].squeeze()

    if np.iscomplexobj(image_data):
        image_data = np.abs(image_data) #we, for now, care only about the magnitude

    if bool(pth_lbl):
        with h5py.File(pth_lbl, 'r', swmr=True) as h5_file:
            label_data = h5_file[dataset_path][:].squeeze()
    
    if slice_index not in [None, -1]:
        image_data = image_data[..., slice_index, :, :].squeeze()
        if bool(pth_lbl):
            label_data = label_data[..., slice_index, :, :].squeeze()

    # Check if the last 2 dimensions are not identical and pad to make it a perfect square
    height, width = image_data.shape[-2], image_data.shape[-1]
    if height != width:
        # Calculate padding needed to make it square
        max_dim = max(height, width)
        pad_height = max_dim - height
        pad_width = max_dim - width
        
        # Apply padding to the last two dimensions
        if len(image_data.shape) == 2:
            # 2D image
            image_data = np.pad(image_data, 
                                ((pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)
            if bool(pth_lbl):
                label_data = np.pad(label_data, 
                                    ((pad_height//2, pad_height - pad_height//2), 
                                    (pad_width//2, pad_width - pad_width//2)), 
                                    mode='constant', constant_values=0)
        elif len(image_data.shape) == 3:
            # 3D image (multiple slices)
            image_data = np.pad(image_data, 
                                ((0, 0), 
                                (pad_height//2, pad_height - pad_height//2), 
                                (pad_width//2, pad_width - pad_width//2)), 
                                mode='constant', constant_values=0)
            if bool(pth_lbl):
                label_data = np.pad(label_data, 
                                    ((0, 0), 
                                    (pad_height//2, pad_height - pad_height//2), 
                                    (pad_width//2, pad_width - pad_width//2)), 
                                    mode='constant', constant_values=0)

    # normalise to [0, 255]    
    if norm_type is not None:
        if norm_type == "minmax": # Method 0: Simple normalisation with Min/Max scaling
            image_data_pre = np.uint8((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255.0)
        elif norm_type == "window":  # Method 1: Window/Level adjustment (like CT/MRI viewing)
            lower_bound = np.percentile(image_data[image_data > 0], window_min_percentile) if np.any(image_data > 0) else 0
            upper_bound = np.percentile(image_data, window_max_percentile)
            image_data_windowed = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = np.uint8((image_data_windowed - lower_bound) / (upper_bound - lower_bound) * 255.0)
        else:
            raise ValueError("Unsupported normalisation type: {}".format(norm_type))
    else:
        image_data_pre = np.uint8(image_data * 255.0)

    if bool(pth_lbl):
        return image_data_pre, label_data, (height, width)
    else:
        return image_data_pre, (height, width)
    
def unpad_arr(padded_arr, prepad_shape):
    """
    Crops a padded array back to its original shape.

    Args:
        padded_arr (torch.Tensor or np.array): The tensor that was padded to be square.
                                      Shape can be (B, (D,) H, W) or (B, C, (D,) H, W).
        prepad_shape (tuple or list): The original shape (height, width) before padding.

    Returns:
        torch.Tensor or np.array: The unpadded (cropped) array.
    """
    original_h, original_w = prepad_shape[-2:]
    padded_h, padded_w = padded_arr.shape[-2:]
    
    pad_h_total = padded_h - original_h
    pad_w_total = padded_w - original_w
    pad_top = pad_h_total // 2
    pad_left = pad_w_total // 2

    unpadded_arr = padded_arr[..., pad_top:pad_top + original_h, pad_left:pad_left + original_w]

    return unpadded_arr

def segment_image(image_data, model_demo, prompt, use_otsu=False, keep_largest_only=False, fill_holes=False):
    segs = []
    model_demo.set_image(image_data)
    for i in range(image_data.shape[0]):
        seg, seg_raw = model_demo.infer(prompt, slice_idx=i, return_raw=True)
        
        if use_otsu:
            thresh = threshold_otsu(seg_raw)
            seg = (seg_raw > thresh).astype(np.uint8)
        
        if keep_largest_only or fill_holes:
            seg_bool = seg.astype(bool)
            
            # Keep only the largest connected component if requested
            if keep_largest_only and np.any(seg_bool):
                labelled = skLabel(seg_bool)
                if labelled.max() > 0:
                    # Find the largest connected component
                    component_sizes = np.bincount(labelled.ravel())
                    component_sizes[0] = 0  # Ignore background
                    largest_component = np.argmax(component_sizes)
                    seg_bool = (labelled == largest_component)
            
            # Fill holes if requested
            if fill_holes and np.any(seg_bool):
                seg_bool = binary_fill_holes(seg_bool)
            
            seg = seg_bool.astype(np.uint8)

        segs.append(seg)
    return np.array(segs)

def post_process_mask(pred_mask, fill_holes=False, keep_largest_component=False):
    """
    Performs post-processing on a multi-class prediction mask.

    This function processes each class (non-zero values) in the input mask
    independently. It can fill holes within each segment and/or keep only the
    largest connected component for each class.

    Args:
        pred_mask (np.ndarray): The input integer mask with unique values for
                                each class (e.g., 0 for background, 1 for
                                class 1, 2 for class 2).
        fill_holes (bool): If True, fills holes within each connected
                           component of each class.
        keep_largest_component (bool): If True, discards all but the largest
                                       connected component for each class.

    Returns:
        np.ndarray: The post-processed mask with the same shape and dtype as
                    the input.
    """
    # Create an empty array to store the final processed mask.
    final_mask = np.zeros_like(pred_mask)

    # Get the unique class IDs, excluding the background (0).
    class_ids = np.unique(pred_mask)
    class_ids = class_ids[class_ids != 0]

    # Process each class ID separately.
    for class_id in class_ids:
        # Create a binary mask for the current class.
        class_mask = (pred_mask == class_id)
        
        # This will hold the processed version of the current class mask.
        processed_class_mask = class_mask.copy()

        # STEP 1: Fill holes for each individual segment.
        if fill_holes:
            # Find all disconnected components for the current class.
            labelled_components, num_components = label(processed_class_mask)
            
            # Create an empty mask to rebuild the hole-filled class mask.
            temp_mask = np.zeros_like(processed_class_mask)
            
            # Iterate through each found component (label 1, 2, ...).
            for i in range(1, num_components + 1):
                component = (labelled_components == i)
                filled_component = binary_fill_holes(component)
                # Add the filled component to our temporary mask.
                temp_mask |= filled_component
            
            processed_class_mask = temp_mask

        # STEP 2: Select the biggest individual segment.
        if keep_largest_component:
            # Label the components of the (potentially hole-filled) mask.
            labelled_components, num_components = label(processed_class_mask)
            
            # Proceed only if there are any components to analyse.
            if num_components > 0:
                component_sizes = np.bincount(labelled_components.ravel())
                
                if len(component_sizes) > 1:
                    largest_component_label = np.argmax(component_sizes[1:]) + 1
                    # Keep only the pixels belonging to the largest component.
                    processed_class_mask = (labelled_components == largest_component_label)
                else:
                    # If no components are found after processing, result in an empty mask.
                    processed_class_mask = np.zeros_like(processed_class_mask)

        final_mask[processed_class_mask] = class_id
        
    return final_mask

def _pepare_prompt_from_text_seg(text_model_demo, text_prompt, text_model_slice, use_otsu_text=False, second_model_mode=None):
    text_seg, seg_raw = text_model_demo.infer(text_prompt, slice_idx=text_model_slice, return_raw=True)
    if use_otsu_text:
        thresh = threshold_otsu(seg_raw)
        text_seg = (seg_raw > thresh).astype(np.uint8)
    text_seg_bool = text_seg.astype(bool)
    # Keep only the largest connected component
    if np.any(text_seg_bool):
        labelled = label(text_seg_bool)
        if labelled.max() > 0:
            # Find the largest connected component
            component_sizes = np.bincount(labelled.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component = np.argmax(component_sizes)
            text_seg_bool = (labelled == largest_component)
    # Fill holes
    if second_model_mode == "point":
        text_seg_bool = binary_fill_holes(text_seg_bool)
    text_seg = text_seg_bool.astype(np.uint8)
    
    if np.any(text_seg):
        y_coords, x_coords = np.where(text_seg > 0)
        if second_model_mode == "point":            
            centre_y = int(np.mean(y_coords))
            centre_x = int(np.mean(x_coords))
            prompt = {"x": centre_x, "y": centre_y}
        elif second_model_mode == "bbox":
            x_min = x_coords.min()
            x_max = x_coords.max()
            y_min = y_coords.min()
            y_max = y_coords.max()
            prompt = {"bbox": np.array([x_min, y_min, x_max, y_max])}
    else:
        prompt = None

    return text_seg, prompt
    
def segment_image_pretext(image_data, text_model_demo, text_prompt, second_model_demo, second_model_mode, text_model_slice=-1, use_otsu_text=False, use_otsu=False, keep_largest_only=False, fill_holes=False):
    text_model_demo.set_image(image_data)
    second_model_demo.set_image(image_data)

    if text_model_slice != -1:
        _, prompt = _pepare_prompt_from_text_seg(text_model_demo, text_prompt, text_model_slice, use_otsu_text=use_otsu_text, second_model_mode=second_model_mode)

    segs = []    
    for i in range(image_data.shape[0]):
        if text_model_slice == -1:
            _, prompt = _pepare_prompt_from_text_seg(text_model_demo, text_prompt, i, use_otsu_text=use_otsu_text, second_model_mode=second_model_mode)

        if prompt is not None:
            seg, seg_raw = second_model_demo.infer(**prompt, slice_idx=i, return_raw=True)
        
            if use_otsu:
                thresh = threshold_otsu(seg_raw)
                seg = (seg_raw > thresh).astype(np.uint8)
        
            if keep_largest_only or fill_holes:
                seg_bool = seg.astype(bool)
                
                # Keep only the largest connected component if requested
                if keep_largest_only and np.any(seg_bool):
                    labelled = label(seg_bool)
                    if labelled.max() > 0:
                        # Find the largest connected component
                        component_sizes = np.bincount(labelled.ravel())
                        component_sizes[0] = 0  # Ignore background
                        largest_component = np.argmax(component_sizes)
                        seg_bool = (labelled == largest_component)
                
                # Fill holes if requested
                if fill_holes and np.any(seg_bool):
                    seg_bool = binary_fill_holes(seg_bool)
                
                seg = seg_bool.astype(np.uint8)
        else:
            seg = np.zeros_like(image_data[i], dtype=np.uint8)

        segs.append(seg)
    return np.array(segs)

def create_overlay(img_data, segs, alpha=0.5):
    overlay_volume_uint8 = np.zeros(img_data.shape + (3,), dtype=np.uint8)  # (slices, H, W, 3)
    cmap = matplotlib.colormaps["tab10"]

    for i in range(img_data.shape[0]):
        img_slice = img_data[i]
        mask_slice = segs[i]
        img_disp = (img_slice - img_slice.min()) / (np.ptp(img_slice) + 1e-8)
        colour_mask = cmap(mask_slice % 10)[..., :3]
        overlay_float = (1 - alpha) * np.stack([img_disp]*3, axis=-1) + alpha * colour_mask * (mask_slice[..., None] > 0)
        overlay_volume_uint8[i] = (overlay_float * 255).astype(np.uint8)

    return overlay_volume_uint8
    

def save_image(image_data, pth, is_RGB=False):
    """
    Save the image data to a file.
    """
    assert (is_RGB and len(image_data.shape) in [3, 4]) or ((not is_RGB) and (len(image_data.shape) in [2, 3])), "Image data must be 2D or 3D, as the function save_image isn't implemented for more. Current shape: {}".format(image_data.shape)

    sitk_image = sitk.GetImageFromArray(image_data, isVector=is_RGB)
    sitk.WriteImage(sitk_image, pth)