import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, \
    edge_preserving_smoothing_3d, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.MO_threshold import MO
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation, ball, disk, erosion, dilation
from aicssegmentation.core.utils import get_middle_frame
from skimage import transform, measure
import h5py
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import warnings
import h5py
from skimage.filters import sobel, scharr, gaussian, median
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, ball
from skimage.measure import label, regionprops, regionprops_table
from aicssegmentation.core.utils import hole_filling
import trackpy as tp
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import napari
from matplotlib.pyplot import cm
from tkinter import simpledialog
from skimage.restoration import rolling_ball
from skimage.restoration import ellipsoid_kernel
from colorama import Fore
from skimage.segmentation import clear_border
import os
from skimage.exposure import rescale_intensity
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def find_tuple(tuple_list, value1, value2):
    """
    Find a specific tuple based on two values in a 4-element tuple.

    Args:
    - tuple_list (list): List of tuples to search through.
    - value1: First value to match in the tuple.
    - value2: Second value to match in the tuple.

    Returns:
    - The first tuple that matches both values.
    - None if no such tuple is found.
    """
    for tpl in tuple_list:
        if tpl[1] == value1 and tpl[2] == value2:
            return tpl[3]
    return None

# Make a function to get middle frame based on segmented area
def get_middle_frame_area(labelled_image_stack):
    max_area = 0
    max_n = 0
    for z in range(labelled_image_stack.shape[0]):
        img_slice = labelled_image_stack[z, :, :]
        area = np.count_nonzero(img_slice)

        if area >= max_area:
            max_area = area
            max_n = z

    return max_n

def get_3dseed_from_stack(bw, stack_shape, hole_min, bg_seed=True):
    from skimage.morphology import remove_small_objects
    out = remove_small_objects(bw > 0, hole_min)

    out1 = label(out)
    stat = regionprops(out1)

    # build the seed for watershed
    seed = np.zeros(stack_shape)
    seed_count = 0
    if bg_seed:
        seed[0, :, :] = 1
        seed_count += 1

    for idx in range(len(stat)):
        pz,py, px = np.round(stat[idx].centroid)
        seed_count += 1
        seed[int(pz), int(py), int(px)] = seed_count

    return seed


# Clean out non-track objects and color by tracks of segmented image stack
def clean_segmentation(table, seg_image):
    original_label = table.label.to_list()
    original_label = [x * 1000 for x in original_label]
    new_label = table.particle.to_list()
    # new_label = [x+1 for x in new_label]

    label_convert = list(zip(original_label, new_label))

    labelled_image = seg_image.copy() * 1000

    for num in np.unique(labelled_image):
        if num != 0 and num not in original_label:
            labelled_image[labelled_image == num] = 0

    for element in label_convert:
        original = element[0]
        new = element[1]
        labelled_image[labelled_image == original] = new

    return labelled_image


# Plot mid section area, volume and Protein binding
# Define a function to plot both section area and protein intensity from panda df
def plotting(df, marker=None):
    static_canvas = FigureCanvas(Figure(figsize=(3, 1)))

    axes = static_canvas.figure.subplots(3, sharex=True)

    # list all remaining tracks
    all_labels = df.particle.unique()

    color = cm.tab20b(np.linspace(0, 1, len(all_labels)))

    axes[0].set_ylabel('Mid Section', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Time(Secs)', fontsize=16, fontweight='bold')
    axes[2].set_ylabel('Normalized Protein Binding', fontsize=16, fontweight='bold')

    num = 0

    for n in all_labels:
        df_subset = df[df.particle == n]
        time_list = df_subset['time'].tolist()
        area_list = df_subset['mid section area'].tolist()
        volume_list = df_subset['volume'].tolist()
        # binding_list = df_subset['intensity ratio'].tolist()
        binding_list = df_subset['normalized intensity ratio'].tolist()
        axes[0].plot(time_list, area_list, color=color[num], label=str(n), marker=marker)
        axes[1].plot(time_list, volume_list, color=color[num], label=str(n), marker=marker)
        axes[2].plot(time_list, binding_list, color=color[num], label=str(n), marker=marker)
        num += 1

    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return static_canvas


# Function to view the segmentation intermediates and final results in Napari Viewer
def Napari_Viewer(structure_img_smooth,intensity_image_stack_raw,intensity_image_stack,segmented_object_image,bw_img,seed_map_list,contour_image_crop,segmented_image_crop,track_points,track_table,x,y,z):
    # Visualize in Napari
    viewer = napari.Viewer()
    viewer.add_image(structure_img_smooth, name='Smooth Image', colormap='green', scale=[1, z, y, x],
                     blending='additive', visible=False)
    viewer.add_image(intensity_image_stack_raw, name='Intensity Image', colormap='red',
                     scale=[1, z, y, x], blending='additive')
    viewer.add_image(intensity_image_stack, name='Intensity Image smooth', colormap='red',
                     scale=[1, z, y, x], blending='additive', visible=False)
    viewer.add_labels(segmented_object_image, name='Segmented Object', scale=[1, z, y, x], blending='additive',
                      visible=False)
    viewer.add_image(bw_img, name='Thresholded', scale=[1, z, y, x], blending='additive', visible=False)
    peaks = np.nonzero(seed_map_list)
    viewer.add_points(np.array(peaks).T, name='peaks', size=5, face_color='red', scale=[1, z, y, x],
                      blending='additive', visible=False)
    viewer.add_image(contour_image_crop, name='contours crop', scale=[1, z, y, x], blending='additive')
    viewer.add_labels(segmented_image_crop, name='Segmented Object Crop', scale=[1, z, y, x],
                      blending='additive', visible=False)
    viewer.add_tracks(track_points, name='tracks', blending='additive',
                      visible=False)

    # Plotting
    canvasobj = plotting(track_table, marker='o')
    viewer.window.add_dock_widget(canvasobj, area='right', name='Analysis Plot')

    napari.run()
## Make the 3D Cell Segmentation Pipeline

# Initiate the cell segmentation class
from skimage import filters


class cell_segment:
    def __init__(self, Time_lapse_image,sigma=1,mask=None):
        self.image = Time_lapse_image.copy()
        self.Time_pts = Time_lapse_image.shape[0]

        self.smooth_param = sigma

        self.bw_img = np.zeros(self.image.shape)
        self.structure_img_smooth = np.zeros(self.image.shape)
        self.segmented_object_image = np.zeros(self.image.shape, dtype=np.uint8)
        self.seed_map_list = np.zeros(self.image.shape, dtype=np.uint8)
        self.mask = mask

    # define a function to apply normalization and smooth on Time lapse images
    def img_norm_smooth(self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        for t in pb:
            pb.set_description("Smooth and background substraction")
            img = self.image[t].copy()
            norm_img = intensity_normalization(img, (2.5, 7.5))
            # Clip data and remove extreme bright speckles
            #vmin, vmax = np.percentile(img, q=(1, 99))

            #clipped_img = rescale_intensity(img, in_range=(vmin, vmax), out_range=np.uint16)

            # self.structure_img_smooth[t] = edge_preserving_smoothing_3d(img, numberOfIterations=5)
            self.structure_img_smooth[t] = image_smoothing_gaussian_3d(norm_img, sigma=self.smooth_param)

            # Use rolling_ball to remove background in Z direction
            background = rolling_ball(self.structure_img_smooth[t], kernel=ellipsoid_kernel((20, 1, 1), 0.2))
            self.structure_img_smooth[t] = self.structure_img_smooth[t] - background

            # define a function to apply Ostu Object thresholding followed by seed-based watershed to each time point

    def threshold_Time(self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        for t in pb:
            pb.set_description("Thresholding and Watershed Segmentation")
            # MO threhodling
            # Check if there is NAN data
            img_sum = np.sum(self.structure_img_smooth[t])
            Nan_check = np.isnan(img_sum)

            # If there is NAN in the data, skip analysis for this image
            if Nan_check == True:
                print("Nan image found")
                continue

            bw, object_for_debug = MO(self.structure_img_smooth[t], global_thresh_method='ave', extra_criteria=True,
                                      object_minArea=4000, return_object=True)

            # Morphological operations to fill holes and remove small/touching objects
            bw = binary_closing(bw, selem=np.ones((4, 4, 4)))
            bw = hole_filling(bw, 1, 40000, fill_2d=True)
            bw = remove_small_objects(bw > 0, min_size=200, connectivity=1, in_place=False)
            bw = clear_border(bw, mask=self.mask)

            self.bw_img[t] = bw

            # Get seed map
            seed = get_3dseed_from_stack(bw, bw.shape, 30, bg_seed=False)

            edge = scharr(self.image[t])
            seg = watershed(edge, markers=label(seed), mask=bw, watershed_line=True)
            seg = clear_border(seg, mask=self.mask)

            seg = remove_small_objects(seg > 0, min_size=300, connectivity=1, in_place=False)
            seg = hole_filling(seg, 1, 1000, fill_2d=True)
            final_seg = label(seg)

            self.segmented_object_image[t] = final_seg
            self.seed_map_list[t] = seed


## Track and quantify nucleus geometry (e.g. volume) and protein binding over time
class cell_tracking:
    def __init__(self, segmented_image_seq, intensity_image_stack, smooth_sigma, x_vol, y_vol, z_vol,time_steps):
        self.labelled_stack = segmented_image_seq
        self.segmented_image_crop = self.labelled_stack.copy()
        self.contour_image_crop = np.zeros(self.labelled_stack.shape)
        self.t = segmented_image_seq.shape[0]
        self.positions_table = None
        self.intensity_image_stack_raw = intensity_image_stack
        self.intensity_image_stack = np.zeros(intensity_image_stack.shape)
        self.smooth_sigma = smooth_sigma
        self.x_vol = x_vol
        self.y_vol = y_vol
        self.z_vol = z_vol
        self.time_steps = time_steps
    # Function to  smooth intesnity image stack
    def intensity_stack_smooth(self):
        # pb = tqdm(range(self.intensity_image_stack.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET))
        pb = tqdm(range(self.intensity_image_stack.shape[0]),
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET))
        for t in pb:
            pb.set_description("Intensity stack smooth")
            img = self.intensity_image_stack_raw[t].copy()
            # img = img[6:-1]
            # self.intensity_image_stack[t] = median(img, ball(self.smooth_sigma))
            # self.intensity_image_stack[t] = image_smoothing_gaussian_slice_by_slice (img, sigma=self.smooth_sigma)

            self.intensity_image_stack[t] = image_smoothing_gaussian_3d(img, sigma=self.smooth_sigma)
            # Use rolling_ball to remove background
            #background = rolling_ball(self.intensity_image_stack[t], kernel=ellipsoid_kernel((20, 1, 1),0.2))
            #self.intensity_image_stack[t] = self.intensity_image_stack[t] - background

    # function to create pandas table of cell attributes without tracking info
    def create_table_regions(self):

        positions = []
        pb = tqdm(range(self.t), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET))
        for n in pb:
            pb.set_description("Create table")
            labelled_slice = self.labelled_stack[n]

            for region in measure.regionprops(labelled_slice):
                position = []

                z_pos = region.centroid[0]
                y_row = region.centroid[1]
                x_col = region.centroid[2]

                volume = region.area * (self.x_vol * self.y_vol * self.z_vol)

                nucleus_image = labelled_slice == region.label
                #mid_z = get_middle_frame_area(nucleus_image)
                mid_z = int(round(z_pos))

                time = find_tuple(self.time_steps,n,mid_z)

                mid_nucleus_image = nucleus_image[mid_z, :, :]

                mid_section_area = max([region.area * (self.x_vol * self.y_vol) for region in
                                        measure.regionprops(label(mid_nucleus_image))])
                # segmented_image_shell= np.logical_xor(erosion(mid_nucleus_image,selem=disk(5)),erosion(mid_nucleus_image, selem=disk(2)))

                # Draw contour
                segmented_image_shell = np.logical_xor(erosion(mid_nucleus_image, selem=disk(1)),
                                                       erosion(mid_nucleus_image, selem=disk(3)))
                bg_segmented_image_shell = mid_nucleus_image

                bg_segmented_image_shell = np.logical_xor(erosion(mid_nucleus_image, selem=disk(5)),
                                                          erosion(mid_nucleus_image, selem=disk(7)))

                if True not in bg_segmented_image_shell:
                    bg_segmented_image_shell = np.zeros(segmented_image_shell.shape, dtype=bool)
                    bg_segmented_image_shell[int(y_row),int(x_col)] = True
                #bg_segmented_image_shell = np.zeros(segmented_image_shell.shape, dtype=bool)
                #bg_segmented_image_shell[int(y_row),int(x_col)] = True
                # bg_segmented_image_shell = np.logical_xor(dilation(mid_nucleus_image, selem=disk(4)), dilation(mid_nucleus_image, selem=disk(7)))

                labels = region.label
                intensity_single = self.intensity_image_stack[n]
                intensity_image = intensity_single[mid_z]
                intensity_median = np.median(intensity_image[segmented_image_shell == True])

                intensity_background = np.median(intensity_image[bg_segmented_image_shell == True])
                # intensity_background = np.median(intensity_image[int(y_row),int(x_col)])

                intensity_median_ratio = intensity_median / intensity_background

                #Convert X,Y,Z from pixels to microns
                z_pos_um = z_pos * self.z_vol
                y_row_um = y_row * self.y_vol
                x_col_um = x_col * self.x_vol

                position.append(x_col)
                position.append(y_row)
                position.append(z_pos)
                position.append(x_col_um)
                position.append(y_row_um)
                position.append(z_pos_um)
                position.append(int(n))
                position.append(time)
                position.append(labels)

                position.append(volume)
                position.append(mid_section_area)
                position.append(intensity_median)
                position.append(intensity_median_ratio)

                positions.append(position)

        self.positions_table = DataFrame(positions,
                                         columns=['x', 'y', 'z','x_micron','y_micron','z_micron', "frame", 'time','label', 'volume', 'mid section area',
                                                  'intensity', 'intensity ratio'])

    # function to track subsequent frame
    def tracking(self, s_range=120, stop=0.95, step=0.99, gap=8, pos=['x_micron', 'y_micron', 'z_micron']):
        self.track_table = tp.link_df(self.positions_table, s_range, adaptive_stop=stop, adaptive_step=step, memory=gap,
                                      pos_columns=pos)
        print(self.track_table)
        # Filter out tracks with low number of frames. Add 1 to particle number to avoid 0.
        #self.track_table = tp.filter_stubs(self.track_table, self.t - 8)
        self.track_table = tp.filter_stubs(self.track_table, self.t-8)
        self.track_table['particle'] = self.track_table['particle'] + 1

        # Extract track points for visualization
        track_df = self.track_table[['particle', 'frame', 'z_micron', 'y_micron', 'x_micron']]
        track_df.index.names = ['Data']
        track_df.sort_values(by=['particle', 'frame'], inplace=True)
        self.track_points = track_df.values

    # function to crop out broken frame based on tracks (only on segmentation image)
    def crop_segmentation(self):
        tracks = np.unique(self.track_points[:, 0])

        remain_track_list = []
        for tk in tracks:
            remain_tracks = self.track_points[self.track_points[:, 0] == tk][:, 1]
            remain_track_list.append(remain_tracks)

        print(remain_track_list)
        # Find all frames (union) among different tracks
        if len(remain_track_list) > 1:
            union_tracks = set.union(*map(set, remain_track_list))
            union_tracks = list(map(int, union_tracks))
        elif len(remain_track_list) == 1:
            union_tracks = remain_track_list[0]
            union_tracks = list(map(int, union_tracks))
        else:
            union_tracks = list(np.arange(self.t))

        print(union_tracks)
        # loop through all time, only keep objects that are in tracks
        pb = tqdm(range(self.t), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        for t in pb:
            pb.set_description("Remove broken objects from segmentation")
            if t in union_tracks:
                table_t = self.track_table.loc[self.track_table.frame == t]
                seg_t = self.labelled_stack[t]
                self.segmented_image_crop[t] = clean_segmentation(table_t, seg_t)

            else:
                if t == 0:
                    table_t = self.track_table.loc[self.track_table.frame == min(union_tracks)]
                    seg_t = self.labelled_stack[min(union_tracks)]
                    self.segmented_image_crop[t] = clean_segmentation(table_t, seg_t)

                else:
                    self.segmented_image_crop[t] = self.segmented_image_crop[t - 1]

    # function to crop out broken frame according to tracks (on mid plane contours)
    def crop_contour(self):
        # loop through all time, only keep objects that are in tracks
        pb = tqdm(range(self.t), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET))
        for t in pb:
            pb.set_description("Remove broken object from mid contour")
            #slice_t = self.segmented_image_crop[t]
            slice_t = self.labelled_stack[t]
            for region in regionprops(slice_t):
                z_pos = region.centroid[0]
                y_row = region.centroid[1]
                x_col = region.centroid[2]

                nucleus_image = slice_t == region.label
                #mid_z = get_middle_frame_area(nucleus_image)
                mid_z = int(round(z_pos))
                mid_nucleus_image = nucleus_image[mid_z, :, :]

                # Draw contours

                segmented_image_shell = np.logical_xor(erosion(mid_nucleus_image, selem=disk(1)),
                                                       erosion(mid_nucleus_image, selem=disk(3)))

                #bg_segmented_image_shell = np.zeros(segmented_image_shell.shape, dtype=bool)
                #bg_segmented_image_shell[int(y_row), int(x_col)] = True

                bg_segmented_image_shell = np.logical_xor(erosion(mid_nucleus_image, selem=disk(5)),
                                                          erosion(mid_nucleus_image, selem=disk(7)))

                if True not in bg_segmented_image_shell:
                    bg_segmented_image_shell = np.zeros(segmented_image_shell.shape, dtype=bool)
                    bg_segmented_image_shell[int(y_row), int(x_col)] = True

                #bg_segemnted_image_shell = 1
                # bg_segmented_image_shell = np.logical_xor(dilation(mid_nucleus_image, selem=disk(4)), dilation(mid_nucleus_image, selem=disk(7)))

                self.contour_image_crop[t, mid_z, :, :] += segmented_image_shell
                self.contour_image_crop[t, mid_z, :, :] += bg_segmented_image_shell
                # self.contour_image_crop[t,mid_z,int(y_row),int(x_col)] = 1

    # function to normalize binding intensity to 1st frame
    def binding_normalize(self):
        self.track_table['normalized intensity ratio'] = self.track_table['intensity ratio']

        all_labels = self.track_table.particle.unique()

        for label in all_labels:
            norm_value = self.track_table.loc[self.track_table.particle == label, 'intensity ratio'].iloc[0]
            self.track_table.loc[self.track_table.particle == label, 'normalized intensity ratio'] = \
            self.track_table.loc[self.track_table.particle == label, 'normalized intensity ratio'] / norm_value


