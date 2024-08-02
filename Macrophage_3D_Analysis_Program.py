from Macrophage_3D_Analysis_Module import *

# Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

# Loading Data, loading Visualization Platform and make some background functions
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'), ('Image files', '.hdf5')]

filename = filedialog.askopenfilenames(parent=root, title='Please Select a File', filetypes=my_filetypes)[0]
print(filename)

'''
from aicsimageio import AICSImage
img = AICSImage(filename)
ER_image = img.get_image_data("TZYX",C=2)
'''

f = h5py.File(filename, 'r')

print(f['561 Channel'].shape)
# Ask the users to define the range of timepoints they want to process
time_answer = simpledialog.askinteger("Input", "What is the last frame point you want to process? ", parent=root, minvalue=1, maxvalue=f['561 Channel'].shape[0])

print(time_answer)
seg_image = f['561 Channel'][:time_answer]
cpla2_intensity_image = f['561 Channel'][:time_answer]

try:
    x_pixel, y_pixel, z_step = f['voxel_info']

except:
    voxel_info = simpledialog.askstring("Question", "Please specify voxel info (z,y,z)")
    x_pixel, y_pixel, z_step = [float(voxel) for voxel in voxel_info.split(',')]

timestep_list = f['Timesteps']

# Make a mask to clear border
from skimage.segmentation import clear_border

mask = np.ones(seg_image[0].shape)

for n in range(mask.shape[0]):
    im_slice = mask[n, :, :]
    im_slice[0, :] = 0
    im_slice[:, 0] = 0
    im_slice[:, -1] = 0
    im_slice[-1, :] = 0

# mask[0,:,:] = 0
# mask[-1,:,:] = 0
mask = mask.astype(bool)

# Smooth and 3D Watershed Segmentation
track = cell_segment(seg_image, mask=mask)
track.img_norm_smooth()
track.threshold_Time()

# Tracking and Quantification
Segment_tracker = cell_tracking(track.segmented_object_image, cpla2_intensity_image, smooth_sigma=1, x_vol=x_pixel,
                                y_vol=y_pixel, z_vol=z_step,time_steps=timestep_list)
Segment_tracker.intensity_stack_smooth()
Segment_tracker.create_table_regions()
print(Segment_tracker.positions_table)
Segment_tracker.tracking()
print(Segment_tracker.track_table)
Segment_tracker.crop_segmentation()
Segment_tracker.crop_contour()
Segment_tracker.binding_normalize()

# Visualize in Napari
Napari_Viewer(track.structure_img_smooth, Segment_tracker.intensity_image_stack_raw,
              Segment_tracker.intensity_image_stack, track.segmented_object_image, track.bw_img, track.seed_map_list,
              Segment_tracker.contour_image_crop, Segment_tracker.segmented_image_crop, Segment_tracker.track_points,
              Segment_tracker.track_table,x=x_pixel,y=y_pixel,z=z_step)

track_table = Segment_tracker.track_table

# Ask whether to delete some measurement due to programming error
del_answer = messagebox.askyesnocancel("Question", "Do you want to delete some measurements?")

while del_answer:
    delete_answer = simpledialog.askstring("Input", "Which numbers do you want to delete? ", parent=root)

    delete_list = list(map(int, delete_answer.split(',')))

    # Delete selected tracks
    track_table = track_table.loc[~track_table.apply(lambda x: x.particle in delete_list, axis=1)]

    # Visualize in Napari again for double check
    Napari_Viewer(track.structure_img_smooth, Segment_tracker.intensity_image_stack_raw,
                  Segment_tracker.intensity_image_stack, track.segmented_object_image, track.bw_img,
                  track.seed_map_list,
                  Segment_tracker.contour_image_crop, Segment_tracker.segmented_image_crop,
                  Segment_tracker.track_points,
                  track_table,x=x_pixel,y=y_pixel,z=z_step)

    del_answer = messagebox.askyesnocancel("Question", "Do you want to delete more measurements?")

# Save segmentation/tracking and analysis result
File_save_names = '.'.join(filename.split(".")[:-1])

# Save data to hdf5
seg_save_name = '{File_Name}_segmentation_result.hdf5'.format(File_Name=File_save_names)

with h5py.File(seg_save_name, "w") as f:
    f.create_dataset('Contour Image', data=Segment_tracker.contour_image_crop, compression='gzip')
    f.create_dataset('Segmented Object', data=Segment_tracker.segmented_image_crop, compression='gzip')
    f.create_dataset('thresholded image', data=track.bw_img, compression='gzip')
    f.create_dataset('Seed Map', data=track.seed_map_list, compression='gzip')

# Save quantification to csv
table_save_name = '{File_Name}_result.csv'.format(File_Name=File_save_names)

track_table.to_csv(table_save_name, index=False)
