import numpy as np
import torch
from torch.utils import data
import glob
import torch.nn.functional as F
import tifffile

def indexGenerate(is_global_coordinates, x_start, y_start, p, size):
    """
    input: x_start, y_start: the starting pixel index for each lens in the original measurement
           p: the cropped measurement size for each lens
           size: the cropped measurement size for each lens
    output: final: [p, p, 4], the 4 channels are (x_orig_norm, y_orig_norm, x_patch_norm, y_patch_norm)
           x_orig_norm, y_orig_norm: normalized original measurement coordinates
           x_patch_norm, y_patch_norm: normalized patch coordinates
    """
    ####### with global coordinates ########
    if is_global_coordinates:
        # normalized index for the original measurement --> normalized global index
        xs = torch.linspace(x_start, x_start + p - 1, steps=p)
        ys = torch.linspace(y_start, y_start + p - 1, steps=p)
        y_orig, x_orig = torch.meshgrid(ys, xs, indexing='ij')
        # 360 and 892 are the starting pixel index for the original measurement --> based on SVFourierNet
        x_orig_norm = (x_orig - 360) / 4560 # something wrong here: 4560 - 360 = 4200
        y_orig_norm = (y_orig - 892) / 5096 # something wrong here: 5096 - 892 = 4204
        
        # normalized index for the cropped patch --> normalized local index for single lens measurement
        xs_patch = torch.linspace(0, size - 1, steps=p)
        ys_patch = torch.linspace(0, size - 1, steps=p)
        y_patch, x_patch = torch.meshgrid(ys_patch, xs_patch, indexing='ij')
        x_patch_norm = x_patch / size
        y_patch_norm = y_patch / size

        # stack the image to [p, p, 4]
        final = torch.stack((x_orig_norm, y_orig_norm, x_patch_norm, y_patch_norm), dim=2)
    else:
        ####### local coordinates for 2d reconstruction ########
        xs_patch = torch.linspace(0, size - 1, steps=p)
        ys_patch = torch.linspace(0, size - 1, steps=p)
        y_patch, x_patch = torch.meshgrid(ys_patch, xs_patch, indexing='ij')
        x_patch_norm = x_patch / size
        y_patch_norm = y_patch / size
        # stack the image to [p, p, 4]
        final = torch.stack((x_patch_norm, y_patch_norm), dim=2)
    return final

class CM2Dataset(data.Dataset):
    def __init__(self, dir_data, transform=None):
        self.dir_data = dir_data
        self.transform = transform
        self.num_stacks = len(glob.glob(self.dir_data + '/meas_*.tif'))//3  # number of measurements used for training

    def __getitem__(self, index):
        # load the measurement
        meas = tifffile.imread(f"{self.dir_data}/meas_{index*3+1}.tif").astype(np.float32)
        mmax = meas.max()
        if mmax > 0: meas /= mmax

        demix = tifffile.imread(f"{self.dir_data}/demix_{index*3+1}.tif").astype(np.float32)
        if demix.ndim == 3 and demix.shape[0] == 9:
            pass
        elif demix.ndim == 3 and demix.shape[2] == 9:
            demix = np.transpose(demix, (2,0,1))
        else:
            raise ValueError(f"demix shape bad: {demix.shape}")
        dmax = demix.max()
        if dmax > 0: demix /= dmax

        gt = tifffile.imread(f"{self.dir_data}/gt_{index*3+1}.tif").astype(np.float32)
        gmax = gt.max()
        if gmax > 0: gt /= gmax

        data_dict = {'gt': gt, 'meas': meas, 'demix': demix}
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        return self.num_stacks


class Crop(object):
    """crop the measurement into 9 views"""
    def __init__(self,  lens_centers, is_global_coordinates=False, tot_len=2400, tmp_pad=900):
        self.lens_centers = lens_centers
        self.tot_len = tot_len
        self.tmp_pad = tmp_pad
        self.is_global_coordinates = is_global_coordinates

        # Local normalized patch coords in [0,1)
        xs = torch.arange(tot_len, dtype=torch.float32)
        ys = torch.arange(tot_len, dtype=torch.float32)
        self.x_patch = (xs / tot_len).view(1, tot_len).expand(tot_len, tot_len)
        self.y_patch = (ys / tot_len).view(tot_len, 1).expand(tot_len, tot_len)

       # Global domain (FIRST valid pixel, EXCLUSIVE end)
        self.X0, self.Y0 = 360.0, 892.0
        self.X1, self.Y1 = 4560.0, 5096.0
        self.W = self.X1 - self.X0   # 4200
        self.H = self.Y1 - self.Y0   # 4204

        # Reusable 1D base index [0..p-1]
        self.base = torch.arange(tot_len, dtype=torch.float32)


    def __call__(self, data):
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        meas = torch.from_numpy(meas)
        meas = F.pad(meas, (self.tmp_pad, self.tmp_pad, self.tmp_pad, self.tmp_pad), 'constant', 0)

        # cropping
        meas_crops = []
        index_list = []
        
        # here need the change for 3D cropping? since it is not the principal ray for each lens?
        for loc in self.lens_centers:
            x_center, y_center = loc
            x_start = x_center - (self.tot_len // 2) + self.tmp_pad # start measurement global index for each lens
            x_end = x_center + (self.tot_len // 2) + self.tmp_pad
            y_start = y_center - (self.tot_len // 2) + self.tmp_pad
            y_end = y_center + (self.tot_len // 2) + self.tmp_pad

            meas_crop = meas[x_start:x_end,y_start:y_end]
            meas_crops.append(meas_crop)

            # Build indices
            if self.is_global_coordinates:
                # # Convert to FULL-IMAGE (pre-padding) starts
                # x_start_full = x_start - self.tmp_pad
                # y_start_full = y_start - self.tmp_pad
                x1d = (x_start + self.base - self.X0) / self.W  # [p]
                y1d = (y_start + self.base - self.Y0) / self.H  # [p]
                x_orig = x1d.view(1, self.tot_len).expand(self.tot_len, self.tot_len)
                y_orig = y1d.view(self.tot_len, 1).expand(self.tot_len, self.tot_len)

                idx = torch.stack((x_orig, y_orig, self.x_patch, self.y_patch), dim=2)  # [H,W,4]
            else:
                idx = torch.stack((self.x_patch, self.y_patch), dim=2)  # [H,W,2]

            index_list.append(idx)

        meas_crops = torch.stack(meas_crops, dim=0)  # [9, H, W]
        index_list = torch.stack(index_list, dim=0)  # [9, H, W, 2]

        # demix is [9, H, W]

        data = {'gt': gt, 'meas': meas_crops, 'demix': demix, 'index': index_list}
        return data
    
class Noisecm2(object):
    """CPU-side noise; does NOT touch index"""
    def __call__(self, data):
        gt, meas, demix, index = data['gt'], data['meas'], data['demix'], data['index']
        amin, amax = 7.8109e-5, 9.6636e-5
        bmin, bmax = 1.3836e-8, 1.1204e-7
        a = np.random.rand() * (amax - amin) + amin
        b = np.random.rand() * (bmax - bmin) + bmin
        # vectorized noise on torch tensor
        noise = torch.sqrt(a * meas + b) * torch.randn_like(meas)
        meas = meas + noise
        return {'gt': gt, 'meas': meas, 'demix': demix, 'index': index}

class ToTensorcm2(object):
    """numpy array to Tensors。"""
    def __call__(self, data):
        gt, meas, demix, index = data['gt'], data['meas'], data['demix'], data['index']
        gt = torch.from_numpy(gt).float().unsqueeze(0)  # [H, W]
        demix = torch.from_numpy(demix).float()
        meas  = meas.float()  # [9, H, W]
        index = index.float()  # [9, H, W, 2]
        return {'gt': gt,  # [1, H, W]
                'meas': meas,  # [9, H, W]
                'demix': demix,  # [9, H, W]
                'index': index
                }

class Subset(data.Dataset):
    def __init__(self, dataset, isVal, patch_size=480, stride=240):
        """
        creating patch based dataset for training and validation
        """
        self.dataset = dataset
        self.isVal = isVal
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = 2400

        if not isVal:
            # calculate the number of patches per image
            self.patches_per_row = (self.image_size - self.patch_size) // self.stride + 1
            self.patches_per_image = self.patches_per_row * self.patches_per_row
            self.total_patches = len(self.dataset) * self.patches_per_image
        else:
            # for validation, we do not crop patches
            self.total_patches = len(self.dataset)

    def __getitem__(self, index):
        if self.isVal:
            data = self.dataset[index]
            return data
        else:
            # get full measurement index and patch index
            stack_index = index // self.patches_per_image
            patch_index = index % self.patches_per_image

            data = self.dataset[stack_index]
            gt, meas, demix, index_list = data['gt'], data['meas'], data['demix'], data['index']

            # calculate the patch location, 2d coordinates of the top-left corner
            row = patch_index // self.patches_per_row
            col = patch_index % self.patches_per_row
            start_y = row * self.stride
            start_x = col * self.stride

            # get gt patch
            gt_patch = gt[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

            # get meas and demix patch
            meas_patch = meas[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
            demix_patch = demix[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]

            # get index_list patch
            index_patch = index_list[:, start_y:start_y + self.patch_size, start_x:start_x + self.patch_size, :]

            # return patch
            data = {'gt': gt_patch, 'meas': meas_patch, 'demix': demix_patch, 'index': index_patch}
            return data

    def __len__(self):
        if self.isVal:
            return self.dataset.__len__()
        else:
            return self.total_patches
