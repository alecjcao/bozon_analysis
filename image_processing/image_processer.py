import numpy as np
from numba import njit
from scipy.sparse.linalg import lsqr
from scipy import sparse
from scipy.optimize import minimize
import logging
from image_processing.point_selector import PointSelector
from image_processing.crop_selector import CropSelector
from image_processing.chimera_parser import get_target_array, check_rearrangement_in_master_script

IMAGE_SIZE = (301, 301)
CROP_SIZE = (210, 200)
LOW_PASS = 6

import platform
name = platform.node()
if name == 'GLaDOS':
    POINTS_FILE = 'A:\\Analysis_Code\\lattice_mask_pts.npy'
elif name == 'burrito':
    POINTS_FILE = '/mnt/heap/Analysis_Code/lattice_mask_pts.npy'
else:
    POINTS_FILE = 'A:\\heap\\Analysis_Code\\lattice_mask_pts.npy'
POINTS = np.load(POINTS_FILE).reshape((48,48,2))
POINTS = POINTS - POINTS[0,0]
DXIDXL = POINTS[0,1,1] - POINTS[0,0,1]
DYIDYL = POINTS[1,0,0] - POINTS[0,0,0]
DXIDYL = POINTS[1,0,1] - POINTS[0,0,1]
DYIDXL = POINTS[0,1,0] - POINTS[0,0,0]
ALL_POINTS = POINTS.reshape(48*48,2)
LOAD_POINTS = POINTS[::2,::3].reshape(24*16,2)
# ALL_POINTS_EXTENDED

class ImageProcesser:
    def __init__(self):
        self._sigma = 1.4
        self._kernel_size = 9
        self.kernel = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)

        self._crop_enabled = True
        self._convolution_enabled = False
        self._all_sites_enabled = False

        self.roi = [50, 50+CROP_SIZE[0], 36, 36+CROP_SIZE[1]]
        
        self.offset = np.array([0, 0])
        self.offset_crop = np.array([0, 0])

    #### PROPERTIES ####

    @property
    def crop_enabled(self):
        return self._crop_enabled
    
    @crop_enabled.setter
    def crop_enabled(self, new_state):
        self._crop_enabled = new_state

    @property
    def convolution_enabled(self):
        return self._convolution_enabled
    
    @convolution_enabled.setter
    def convolution_enabled(self, new_state):
        self._convolution_enabled = new_state

    @property
    def all_sites_enabled(self):
        return self._all_sites_enabled
    
    @all_sites_enabled.setter
    def all_sites_enabled(self, new_state):
        self._all_sites_enabled = new_state

    @property
    def kernel_size(self):
        return self._kernel_size
    
    @kernel_size.setter
    def kernel_size(self, new_size):
        try:
            if int(new_size) % 2 == 1:
                self._kernel_size = int(new_size)
                self.kernel = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
            else:
                raise ValueError("Kernel size must be an odd integer.")
        except ValueError:
            raise ValueError("Kernel size must an odd integer.")
        
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, new_sigma):
        try:
            if float(new_sigma) > 0:
                self._sigma = float(new_sigma)
            else:
                raise ValueError("Sigma must be a positive number.")
        except ValueError:
            raise ValueError("Sigma must be a positive number.")
    
    #### HELPER METHODS ####
        
    @njit
    def get_gauss(self, subpixel_shift):
        """
        Returns a 2D Gaussian evaluated over a kernel. 
        """
        gauss_x = np.exp(-(self.kernel-subpixel_shift[1])**2/(2*self.sigma**2))
        gauss_y = np.exp(-(self.kernel-subpixel_shift[0])**2/(2*self.sigma**2))
        return gauss_y[:,None]*gauss_x[None,:]
    
    @njit
    def get_masks(self, centers):
        masks = np.zeros(IMAGE_SIZE, dtype = np.float64)

        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - self.kernel_size//2)
            y_start = int(np.round(center[0]) - self.kernel_size//2)
            x_end = x_start + self.kernel_size
            y_end = y_start + self.kernel_size
            subpixel_shift = center - np.round(center)
            if x_start<0 or y_start<0 or x_end>IMAGE_SIZE[1] or y_end>IMAGE_SIZE[0]:
                continue
            masks[y_start:y_end, x_start:x_end] += self.get_gauss(subpixel_shift)

        return masks
    
    @njit
    def get_counts(self, image, centers):
        counts = np.zeros(centers.shape[0], dtype=np.float64)
        
        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - self.kernel_size//2)
            y_start = int(np.round(center[0]) - self.kernel_size//2)
            x_end = x_start + self.kernel_size
            y_end = y_start + self.kernel_size
            subpixel_shift = center - np.round(center)

            mask = self.get_gauss(subpixel_shift)
            counts[i] = np.sum((image[y_start:y_end, x_start:x_end]*mask))/np.sum(mask)

        return counts
    
    @njit
    def get_convolution_matrix(self, centers):
        convolution_matrix = np.zeros(len(centers)*self.kernel_size**2, dtype = float)
        center_inds = np.zeros(len(centers)*self.kernel_size**2, dtype = float)
        image_inds = np.zeros(len(centers)*self.kernel_size**2, dtype = float)
        delete_inds = []
        
        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - self.kernel_size//2)
            y_start = int(np.round(center[0]) - self.kernel_size//2)
            x_end = x_start + self.kernel_size
            y_end = y_start + self.kernel_size
            
            if x_start<0 or y_start<0 or x_end>IMAGE_SIZE[1] or y_end>IMAGE_SIZE[0]:
                for j in range(i*self.kernel_size**2, (i+1)*self.kernel_size**2):
                    delete_inds.append(j)
                continue
            
            center_inds[i*self.kernel_size**2:(i+1)*self.kernel_size**2] = i
            image_inds[i*self.kernel_size**2:(i+1)*self.kernel_size**2] = (np.arange(y_start, y_end)[:,None]*IMAGE_SIZE[1] + np.arange(x_start, x_end)[None,:]).flatten()
            
            subpixel_shift = center - np.round(center)
            convolution_matrix[i*self.kernel_size**2:(i+1)*self.kernel_size**2] = self.get_gauss(subpixel_shift).flatten()

        if len(delete_inds)>0:
            image_inds = np.delete(image_inds, delete_inds)
            center_inds = np.delete(center_inds, delete_inds)
            convolution_matrix = np.delete(convolution_matrix, delete_inds)
        return image_inds, center_inds, convolution_matrix

    def mask_weighted_sum(self, image, points):
        mask = self.get_masks(points + self.offset)
        return -np.sum(mask*image)


    #### MAIN IMAGE PROCESSING PROCEDURE ####

    def process_images(self, data):
        ## Unpack data
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = np.array(data['Andor']['Pictures'])
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        ## Remove anomalous pixels
        anomalous_pixels = images>10000
        images[anomalous_pixels] = np.nan
        logging.warning(f'Removed {len(anomalous_pixels)} anomalous pixels.')

        ## Background subtraction
        bg_row = np.nanmean(images[:, 0, :30], axis = (0, 1))
        images = images - bg_row[None, None, None, :]
        images[np.isnan(images)] = 0

        ## Crop images
        if self.crop_enabled:
            images =  images[:, :, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        ## Pre-detect tweezer position jumps from Chimera data
        try:
            if data['Gmoog-Parameters']['auto-offset-active']:
                xoff = np.array(data['Tweezer-auto-offset']['x-auto-offset-MHz'])
                yoff = np.array(data['Tweezer-auto-offset']['y-auto-offset-MHz'])
                xjumps = np.sign(np.diff(xoff)) * (np.abs(np.diff(xoff))>1/2)
                yjumps = np.sign(np.diff(yoff)) * (np.abs(np.diff(yoff))>1/2)
            else:
                xjumps = np.zeros(images.shape[0]-1)
                yjumps = np.zeros(images.shape[0]-1)
        except KeyError:
            logging.warning('Key error accessing auto-offset data.')
            xjumps = np.zeros(images.shape[0]-1)
            yjumps = np.zeros(images.shape[0]-1)

        ## Determine whether rearrangement was executed
        if not self.all_sites_enabled:
            rerng_in_master = check_rearrangement_in_master_script(data['Master-Parameters']['Master-Script'][:])
            if 'rearranger_active' in data['Gmoog-Parameters'].keys():
                rerng_active = bool(data['Gmoog-Parameters']['rearranger_active'][0])
            else:
                rerng_active = True
            if 'enable_rearrange' in data['Gmoog-Parameters']['Variable'].keys():
                rerng_enabled = bool(data['Gmoog-Parameters']['Variable']['enable_rearrange'][0])
            else:
                rerng_enabled = True
            rearranged = rerng_in_master and rerng_active and rerng_enabled
            if rearranged:
                target_points = POINTS[get_target_array(data['Gmoog-Parameters']['gmoog_script']),:]
            else:
                target_points = LOAD_POINTS
        else:
            target_points = ALL_POINTS


        ## Fit array offsets from first image and then get site counts for each image
        fitted_shifts = np.zeros((images.shape[0], 2), dtype = float)
        counts = np.zeros((images.shape[0], target_points.shape[0]), dtype = float)
        for i in range(images.shape[0]):
            res = minimize(self.mask_weighted_sum, self.offset, args = (images[i,0], LOAD_POINTS), method = 'Nelder-Mead',
                            options = {'xatol': 1e-2, 'fatol': 10, 'maxiter': 30})
            if i>0 and i<images.shape[0]-1:
                self.offset = ((LOW_PASS-1)*self.offset + res.x)/LOW_PASS + + np.array([yjumps[i-1]*DYIDYL, xjumps[i-1]*DXIDXL])

            for j in range(pics_per_rep):
                if j==0 and not self.all_sites_enabled:
                    self.get_counts(images[i,j], LOAD_POINTS)
                else:
                    self.get_counts(images[i,j], target_points)
            
        # return 'hi'
    
    #### 
    
    def select_crop_region(self, data, parent):
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = np.array(data['Andor']['Pictures'])
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image = np.mean(images[:,0], axis = 0)
        
        point = None
        popup = CropSelector(image, parent, "Select crop roi (click top left corner)")
        if popup.exec_():  # If user confirms
            point = popup.selected_point

        if point:
            self.roi = [point[1], point[1]+CROP_SIZE[0], point[0], point[0]+CROP_SIZE[1]]
        else:
            logging.warning("No crop region selected.")

    def select_offset(self, data, parent):
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = np.array(data['Andor']['Pictures'])
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image = np.mean(images[:,0], axis = 0)

        if self.crop_enabled:
            image = image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        
        point = None
        popup = PointSelector(image, parent, "Click top left atom")
        if popup.exec_():  # If user confirms
            point = popup.selected_point

        if point:
            if self.crop_enabled:
                self.offset_crop = [point[1], point[0]]
            else:
                self.offset = [point[1], point[0]]
        else:
            logging.warning("No crop region selected.")