import numpy as np
from numba import njit

from scipy.sparse.linalg import lsqr
from scipy import sparse
from scipy.optimize import minimize
from scipy.signal import savgol_filter, find_peaks

import datetime
import xarray as xr

import logging

from image_processing.point_selector import PointSelector
from image_processing.crop_selector import CropSelector
from image_processing.chimera_parser import get_target_array, check_rearrangement_in_master_script

from skimage.filters import threshold_minimum
import time

IMAGE_SIZE = (301, 301)
TOTAL_PIXELS = np.prod(IMAGE_SIZE)
CROP_SIZE = (210, 200)
CROP_PIXELS = np.prod(CROP_SIZE)
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

SIGMA = 1.4
KERNEL_SIZE = 9
KERNEL = np.arange(-(KERNEL_SIZE//2), KERNEL_SIZE//2+1)

@njit
def get_gauss(subpixel_shift):
    """
    Returns a 2D Gaussian evaluated over a kernel. 
    """
    gauss_x = np.exp(-(KERNEL-subpixel_shift[1])**2/(2*SIGMA**2))
    gauss_y = np.exp(-(KERNEL-subpixel_shift[0])**2/(2*SIGMA**2))
    return gauss_y[:,None]*gauss_x[None,:]

class ImageProcessor:
    def __init__(self, data_handler, figure):
        # self._sigma = 1.4
        # self._kernel_size = 9
        # self.kernel = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)

        self._crop_enabled = True
        self._convolution_enabled = False
        self._all_sites_enabled = False

        self.roi = [50, 50+CROP_SIZE[0], 36, 36+CROP_SIZE[1]]
        
        self.offset = np.array([0, 0])
        self.offset_switch = np.array([0, 0])  # holds value for switching between crop and uncropped mode

        self._default_threshold = 10

        self.data_handler = data_handler
        self.figure = figure

        ImageProcessor.get_masks(LOAD_POINTS, IMAGE_SIZE)
        ImageProcessor.get_counts(np.zeros(IMAGE_SIZE), LOAD_POINTS)
        ImageProcessor.get_convolution_matrix(LOAD_POINTS, True)

    #### PROPERTIES ####

    @property
    def crop_enabled(self):
        return self._crop_enabled
    
    @crop_enabled.setter
    def crop_enabled(self, new_state):
        if new_state != self._crop_enabled:
            self._crop_enabled = new_state
            self.offset_switch, self.offset = self.offset, self.offset_switch

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

    # @property
    # def kernel_size(self):
    #     return self._kernel_size
    
    # @kernel_size.setter
    # def kernel_size(self, new_size):
    #     try:
    #         if int(new_size) % 2 == 1:
    #             self._kernel_size = int(new_size)
    #             self.kernel = np.arange(-(self.kernel_size//2), self.kernel_size//2+1)
    #         else:
    #             raise ValueError("Kernel size must be an odd integer.")
    #     except ValueError:
    #         raise ValueError("Kernel size must an odd integer.")
        
    # @property
    # def sigma(self):
    #     return self._sigma
    
    # @sigma.setter
    # def sigma(self, new_sigma):
    #     try:
    #         if float(new_sigma) > 0:
    #             self._sigma = float(new_sigma)
    #         else:
    #             raise ValueError("Sigma must be a positive number.")
    #     except ValueError:
    #         raise ValueError("Sigma must be a positive number.")
        
    @property
    def default_threshold(self):
        return self._default_threshold
    
    @default_threshold.setter
    def default_threshold(self, new_threshold):
        try:
            if float(new_threshold) > 0:
                self._default_threshold = float(new_threshold)
            else:
                raise ValueError("Threshold must be a positive number.")
        except ValueError:
            raise ValueError("Threshold must be a positive number.")

    
    #### STATIC HELPER METHODS ####

    @staticmethod
    @njit
    def get_masks(centers, image_shape):
        masks = np.zeros(image_shape)

        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - KERNEL_SIZE//2)
            y_start = int(np.round(center[0]) - KERNEL_SIZE//2)
            x_end = x_start + KERNEL_SIZE
            y_end = y_start + KERNEL_SIZE
            subpixel_shift = center - np.round(center)
            if x_start<0 or y_start<0 or x_end>image_shape[1] or y_end>image_shape[0]:
                continue
            masks[y_start:y_end, x_start:x_end] += get_gauss(subpixel_shift)

        return masks
    
    @staticmethod
    @njit
    def get_counts(image, centers):
        counts = np.zeros(centers.shape[0], dtype=np.float64)
        image_shape = image.shape
        
        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - KERNEL_SIZE//2)
            y_start = int(np.round(center[0]) - KERNEL_SIZE//2)
            x_end = x_start + KERNEL_SIZE
            y_end = y_start + KERNEL_SIZE
            subpixel_shift = center - np.round(center)
            if x_start<0 or y_start<0 or x_end>image_shape[1] or y_end>image_shape[0]:
                continue
            mask = get_gauss(subpixel_shift)
            counts[i] = np.sum((image[y_start:y_end, x_start:x_end]*mask))/np.sum(mask)

        return counts
    
    @staticmethod
    @njit
    def get_convolution_matrix(centers, crop_enabled):
        convolution_matrix = np.zeros(len(centers)*KERNEL_SIZE**2, dtype = float)
        center_inds = np.zeros(len(centers)*KERNEL_SIZE**2, dtype = float)
        image_inds = np.zeros(len(centers)*KERNEL_SIZE**2, dtype = float)
        delete_inds = []

        if crop_enabled:
            image_size = CROP_SIZE
        else:
            image_size = IMAGE_SIZE
        
        for i in range(centers.shape[0]):
            center = centers[i]
            x_start = int(np.round(center[1]) - KERNEL_SIZE//2)
            y_start = int(np.round(center[0]) - KERNEL_SIZE//2)
            x_end = x_start + KERNEL_SIZE
            y_end = y_start + KERNEL_SIZE
            subpixel_shift = center - np.round(center)
            
            out_of_bounds = x_start<0 or y_start<0 or x_end>image_size[1] or y_end>image_size[0]
            if out_of_bounds:
                for j in range(i*KERNEL_SIZE**2, (i+1)*KERNEL_SIZE**2):
                    delete_inds.append(j)
                continue
            
            center_inds[i*KERNEL_SIZE**2:(i+1)*KERNEL_SIZE**2] = i
            image_inds[i*KERNEL_SIZE**2:(i+1)*KERNEL_SIZE**2] = (np.arange(y_start, y_end)[:,None]*image_size[1] + np.arange(x_start, x_end)[None,:]).flatten()
            
            subpixel_shift = center - np.round(center)
            convolution_matrix[i*KERNEL_SIZE**2:(i+1)*KERNEL_SIZE**2] = get_gauss(subpixel_shift).flatten()

        if len(delete_inds)>0:
            image_inds = np.delete(image_inds, delete_inds)
            center_inds = np.delete(center_inds, delete_inds)
            convolution_matrix = np.delete(convolution_matrix, delete_inds)
        return image_inds, center_inds, convolution_matrix

    @staticmethod
    def mask_weighted_sum(offset, image, points):
        mask = ImageProcessor.get_masks(points + offset, image.shape)
        return -np.sum(mask*image)


    #### MAIN IMAGE PROCESSING PROCEDURE ####

    def process_images(self):
        ## Unpack data
        data, images = self.data_handler.get_raw_data()
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        ## Remove anomalous pixels
        anomalous_pixels = images>10000
        if np.any(anomalous_pixels):
            images[anomalous_pixels] = np.nan
            logging.warning(f'Removed {np.sum(anomalous_pixels)} anomalous pixels.')

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
            logging.warning('No auto-offset data.')
            xjumps = np.zeros(images.shape[0]-1)
            yjumps = np.zeros(images.shape[0]-1)

        ## Determine whether rearrangement was executed
        if not self.all_sites_enabled:
            rerng_in_master, trigger_count = check_rearrangement_in_master_script(data['Master-Parameters']['Master-Script'][:],
                data['Master-Parameters']['Functions'])
            if 'rearranger_active' in data['Gmoog-Parameters'].keys():
                rerng_active = bool(data['Gmoog-Parameters']['rearranger_active'][0])
            else:
                rerng_active = True
            if 'enable_rearrange' in data['Master-Parameters']['Variables'].keys():
                rerng_enabled = bool(data['Master-Parameters']['Variables']['enable_rearrange'][0])
            else:
                rerng_enabled = True
            rearranged = rerng_in_master and rerng_active and rerng_enabled
            if rearranged:
                gmoog_script = data['Gmoog-Parameters']['Gmoog-Script'][:]
                target_points = POINTS[get_target_array(gmoog_script, trigger_count),:]
            else:
                target_points = LOAD_POINTS
        else:
            target_points = ALL_POINTS

        ## Fit array offsets from first image and then get site counts for each image
        fitted_shifts = np.zeros((images.shape[0], 2))
        max_num_sites = max(len(LOAD_POINTS), len(target_points))
        counts = np.zeros((pics_per_rep, images.shape[0], max_num_sites))
        for i in range(images.shape[0]):
            res = minimize(ImageProcessor.mask_weighted_sum, self.offset, args = (images[i,0], LOAD_POINTS), method = 'Nelder-Mead',
                            options = {'xatol': 1e-2, 'fatol': 10, 'maxiter': 30})
            if i>0 and i<images.shape[0]-1:
                self.offset = ((LOW_PASS-1)*self.offset + res.x)/LOW_PASS + np.array([yjumps[i-1]*DYIDYL, xjumps[i-1]*DXIDXL])
            fitted_shifts[i] = res.x

            if self.convolution_enabled:
                im_inds, cen_inds, cm_data = ImageProcessor.get_convolution_matrix(target_points + fitted_shifts[i], self.crop_enabled)
                cmat = sparse.coo_matrix((cm_data,(im_inds, cen_inds)), 
                                         shape = (CROP_PIXELS*self.crop_enabled + TOTAL_PIXELS*(1-self.crop_enabled), 
                                                  len(target_points))).tocsc()
            for j in range(pics_per_rep):
                if j==0:
                    counts[j,i,:len(LOAD_POINTS)] = ImageProcessor.get_counts(images[i,j], LOAD_POINTS + fitted_shifts[i])
                else:
                    if self.convolution_enabled:
                        counts[j,i,:len(target_points)] = lsqr(cmat, images[i,j].flatten(), atol = 1e-2, btol = 1e-2, iter_lim = 10)[0] 
                    else:
                        counts[j,i,:len(target_points)] = ImageProcessor.get_counts(images[i,j], target_points + fitted_shifts[i])

        ## Fit thresholds and get site occupations
        detections = np.zeros((pics_per_rep, images.shape[0], max_num_sites), dtype = np.float64)
        thresholds = np.zeros(pics_per_rep, dtype = np.float64)
        for j in range(pics_per_rep):
            detections[j], thresholds[j] = self.threshold_counts(counts[j])

        ## plot image processing results
        self.figure.clf()
        self.figure.suptitle(self.data_handler.date.strftime('%y%m%d') + ' File' + str(self.data_handler.file))
        ax = self.figure.subplots(2,2)
        ax[0,0].imshow(np.mean(images, axis = (0,1)))
        ax[0,0].plot(target_points[:,1]+np.mean(fitted_shifts[:,1]), target_points[:,0]+np.mean(fitted_shifts[:,0]), 'r.', ms = 1)
        ax[0,1].plot(fitted_shifts[:,1], label = 'x')
        ax[0,1].plot(fitted_shifts[:,0], label = 'y')
        ax[0,1].legend()
        for i in range(pics_per_rep):
            h = ax[1,0].hist(counts[i].flatten(), bins = range(-20, 100), alpha = 0.5)
            ax[1,0].axvline(thresholds[i], color = h[-1][-1].get_facecolor())
        ax[1,0].set_yscale('log')
        ax[1,0].set_xlim(-20, 100)
        ax[1,1].plot(np.mean(detections[0], axis = -1), label = 'fill')
        ax[1,1].plot(np.nansum(detections[-1], axis = -1)/np.nansum(detections[-2], axis = -1), label = 'survival')
        ax[1,1].legend()
        ax[1,1].set_ylim(-.05, 1.05)
        self.figure.tight_layout()

        ds = self.export_to_xarray(data, detections, fitted_shifts, target_points)
        self.data_handler.save_processed_dataset(ds)
        self.data_handler.save_image_processing_fig(self.figure)
        
    #### 
    
    def select_crop_region(self, parent):
        data, images = self.data_handler.get_raw_data()
        if not self.crop_enabled:
            logging.warning('Enable crop to select crop roi.')
            return
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image = np.mean(images[:,0], axis = 0)
        
        point = None
        popup = CropSelector(image, parent, "Select crop roi (click top left corner)")
        if popup.exec_():  # If user confirms
            point = popup.selected_point

        if point:
            self.offset += np.array([self.roi[0] - point[1], self.roi[2] - point[0]])
            self.roi = [point[1], point[1]+CROP_SIZE[0], point[0], point[0]+CROP_SIZE[1]]
        else:
            logging.warning("No crop region selected.")

    def select_offset(self, parent):
        data, images = self.data_handler.get_raw_data()
        pics_per_rep = data['Andor']['Pictures-Per-Repetition'][0]
        images = images.reshape((-1, pics_per_rep, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        image = np.mean(images[:,0], axis = 0)

        if self.crop_enabled:
            image = image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        
        point = None
        popup = PointSelector(image, parent, "Click top left atom")
        if popup.exec_():  # If user confirms
            point = popup.selected_point

        if point:
            self.offset = np.array([point[1], point[0]])
            if self.crop_enabled:
                self.offset_switch = np.array([point[1]+self.roi[0], point[0]+self.roi[2]])
            else:
                self.offset_switch = np.array([point[1]-self.roi[0], point[0]-self.roi[2]])
        else:
            logging.warning("No offset selected.")

    def export_to_xarray(self, data, detections, shifts, target_points):
        # prepare global attributes shared across all images
        date = ''.join([x.decode('UTF-8') for x in data['Miscellaneous']['Run-Date']])
        time = ''.join([x.decode('UTF-8') for x in data['Miscellaneous']['Time-Of-Logging']])
        dt = datetime.datetime.strptime(' '.join((
            date, time[:-1])), "%Y-%m-%d %H:%M:%S")

        global_attrs = {
                'experiment_datetime': dt,
                'file_number': self.data_handler.file,
                'target_x': target_points[:, 0],
                'target_y': target_points[:, 1],
            }
        
        # figure out constant vs variable parameters
        variables = data['Master-Parameters']['Variables']
        repetitions = data['Master-Parameters']['Repetitions'][0]
        key_name = []
        key = []
        for name in variables.keys():
            variable = variables[name]
            if variable.attrs['Constant'][0]:
                global_attrs[name] = variable[0]
            else:
                key_name.append(name)
                key.append(np.array(variable))
        if len(key_name) > 0:
            global_attrs['key_names'] = key_name
        else:
            global_attrs['key_names'] = ['No-Variation']
        data_export = []

        for i in range(detections.shape[1]):
            load = detections[0, i, :16*24].astype(bool)
            target_fill = detections[-2, i, :len(target_points)].astype(bool)
            target_second = detections[-1, i, :len(target_points)].astype(bool)
            shiftx, shifty = shifts[i]

            datum = {
                'load': load,
                'fill': target_fill,
                'second': target_second,
                'shiftx': shiftx,
                'shifty': shifty
            }
            datum.update(**global_attrs)

            for j, name in enumerate(key_name):
                if name == 'No-Variation':
                    break
                datum[name] = key[j][i//repetitions]

            data_export.append(datum)

        # aggregate data in a format compatible with xarray
        data_export_xr = {}
        columns = data_export[0].keys()
        for key in columns:
            data_export_xr[key] = {'dims': ['id'], 'data': []}
        data_export_xr['load']['dims'] = ['id', 'ind_load']
        for key in ('fill', 'second', 'target_x', 'target_y'):
            data_export_xr[key]['dims'] = ['id', 'ind']    
        data_export_xr['key_names']['dims'] = ['id', 'variables']
        for datum in data_export:
            for key in columns:
                data_export_xr[key]['data'].append(datum[key])
        
        ds = xr.Dataset.from_dict(data_export_xr)
        return ds
    
    def threshold_counts(self, counts):
        counts_max = np.max(counts)
        counts_min = np.min(counts)
        if counts_max > counts_min:
            counts_scaled = (counts - counts_min) / (counts_max - counts_min) * 255
        else:
            counts_scaled = np.zeros_like(counts)
        try:
            threshold = threshold_minimum(counts_scaled.astype(np.uint8))
            threshold = threshold / 255 * (counts_max - counts_min) + counts_min
        except RuntimeError:
            logging.warning(f"Couldn't find threshold. Using default {self.default_threshold}.")
            threshold = self.default_threshold     
        counts_thresholded = counts > threshold
        return counts_thresholded, threshold