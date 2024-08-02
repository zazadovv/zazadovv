from nd2reader.reader import ND2Reader
import numpy as np
import time
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk 
import matplotlib.pyplot as plt
from skimage import io
import warnings
import h5py
from colorama import Fore
from tqdm import tqdm
import os 
import glob

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

root.directory = filedialog.askdirectory()
Directory_name = root.directory

answer = messagebox.askyesno("Question","Does the image contain multiple z stacks?")


def group_objects_with_ids(lst, group_size):
    grouped_list = []

    for i in range(0, len(lst), group_size):
        primary_id = i // group_size + 1
        primary_group = lst[i:i + group_size]

        # Assign individual IDs to each object within the primary group
        secondary_group = [(obj_id + 1, obj) for obj_id, obj in enumerate(primary_group)]

        grouped_list.append((primary_id, secondary_group))

    return grouped_list


def findDict (d, keyName):
    if not isinstance (d, dict): return None
    if keyName in d: return d [keyName]
    for subdict in d.values ():
        r = findDict (subdict, keyName)
        if r: return r

def Z_Stack_Images_Extractor(Image_sequence, fields_of_view, z_answer):
   Channel_list = Image_Sequence.metadata['channels']
   
   # Check whether the data has one channel or two channels. If only one channel, put GUV_Channel and Protein_Channel = 0
   if len(Channel_list) == 1:
      GUV_Channel = 0
      Protein_Channel = 0

   # Select correct channels for downstream import
   elif len(Channel_list) == 2:
     if 'DsRed' in Channel_list[0] or '561' in Channel_list[0]:
        GUV_Channel = 0
        Protein_Channel = 1

     else:
        GUV_Channel = 1
        Protein_Channel = 0

   else:
      if 'DsRed' in Channel_list[0] or '561' in Channel_list[0]:
         if 'GFP' in Channel_list[0] or '488' in Channel_list[1]:
           GUV_Channel = 0
           Protein_Channel = 1
         
         else:
           GUV_Channel = 0
           Protein_Channel = 2
      
      elif 'DsRed' in Channel_list[1] or '561' in Channel_list[1]:
         if 'GFP' in Channel_list[0] or '488' in Channel_list[0]:
           GUV_Channel = 1
           Protein_Channel = 0
         
         else:
           GUV_Channel = 1
           Protein_Channel = 2
      else:
        if 'GFP' in Channel_list[0] or '488' in Channel_list[0]:
          GUV_Channel = 2
          Protein_Channel = 0
         
        else:
          GUV_Channel = 2
          Protein_Channel = 1

   time_series = Image_Sequence.sizes['t']
   
   if z_answer:
     z_stack = Image_Sequence.sizes['z']
   
   Intensity_Slice = []
   GUV_Slice = []

   n = 0

   for time in range(time_series):

     if z_answer:
       z_stack_images = [] 
       z_stack_Intensity_images = []
       for z_slice in range(z_stack):
          slice = Image_Sequence.get_frame_2D(c=GUV_Channel, t=time, z=z_slice, v=fields_of_view)
          Intensity_slice = Image_Sequence.get_frame_2D(c=Protein_Channel, t=time, z=z_slice, v=fields_of_view)
          z_stack_images.append(slice)
          z_stack_Intensity_images.append(Intensity_slice)
     


       z_stack_images = np.array(z_stack_images)
       z_stack_Intensity_images = np.array(z_stack_Intensity_images)
     
     else:
        z_stack_images = Image_Sequence.get_frame_2D(c=0, t=time, v=fields_of_view)
        z_stack_Intensity_images = Image_Sequence.get_frame_2D(c=1, t=time, v=fields_of_view)

     GUV_Slice.append(z_stack_images)

     Intensity_Slice.append(z_stack_Intensity_images)
     
   GUV_Slice = np.array(GUV_Slice)
   Intensity_Slice = np.array(Intensity_Slice)

   return (GUV_Slice, Intensity_Slice)




os.chdir(Directory_name)

df_filenames = glob.glob('*.nd2' )

# create progress bar
pb = tqdm(range(len(df_filenames)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))

for img_num in pb:
   pb.set_description("Converting nd2 to hdf5 files")
   
   img = df_filenames[img_num]
   
   Image_Sequence = ND2Reader(img)

   #Get z_step and x and y pixels
   z_step = findDict(Image_Sequence.parser._raw_metadata.image_metadata,b'dZStep')
   x_pixel= Image_Sequence.metadata['pixel_microns']
   y_pixel= Image_Sequence.metadata['pixel_microns']

   voxel_info = (x_pixel, y_pixel, z_step)

   print(voxel_info)

   # Get timesteps info
   Timesteps = list(Image_Sequence.timesteps)

   # Grouping every 5 elements
   grouped_timesteps = group_objects_with_ids(Timesteps,Image_Sequence.sizes['z'])

   FOV_list = Image_Sequence.metadata['fields_of_view']


   File_save_names = '.'.join(img.split(".")[:-1])

   for fov in range(len(FOV_list)):

      FOV_time = [fov+1 + i * len(FOV_list) for i in range(Image_Sequence.sizes['t'])]


      grouped_timesteps_fov = [tpl for tpl in grouped_timesteps if tpl[0] in FOV_time]

      timestep_list_fov = []



      for n in range(len(grouped_timesteps_fov)):

         single_stack = grouped_timesteps_fov[n]

         for z in single_stack[1]:

            timestep_list_fov.append((fov, n, z[0] - 1, z[1] / 1000))

      print(timestep_list_fov)

      Channel_561, Channel_488 = Z_Stack_Images_Extractor(Image_Sequence,fields_of_view=fov,z_answer=answer)

      Image_Name='{File_Name}_{num}.hdf5'.format(File_Name = File_save_names, num = fov + 1)

      with h5py.File(Image_Name, "w") as f:
         f.create_dataset('488 Channel', data = Channel_488, compression = 'gzip')
         f.create_dataset('561 Channel', data = Channel_561, compression = 'gzip')
         f.create_dataset('voxel_info', data=voxel_info, compression='gzip')
         f.create_dataset('Timesteps', data=timestep_list_fov, compression='gzip')

'''my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)
'''


# Define a function to convert time series of ND2 images to a numpy list of 
# images (t,z,y,x).





'''

Image_Sequence = ND2Reader(Image_Stack_Path)
FOV_list = Image_Sequence.metadata['fields_of_view']

GUV_Image_list = []
Intensity_list = []

for fov in range(len(FOV_list)):
   GUV_Images, Image_Intensity = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=fov,z_answer=answer)
   GUV_Image_list.append(GUV_Images)
   Intensity_list.append(Image_Intensity)


File_save_names = '.'.join(Image_Stack_Path.split(".")[:-1])

for n in range(len(FOV_list)):
   GUV_Image_Name='{File_Name}_{num}.hdf5'.format(File_Name = File_save_names, num = n + 1)
   
   GUV_Images = GUV_Image_list[n]
   Image_Intensity = Intensity_list[n]

   with h5py.File(GUV_Image_Name, "w") as f:
      f.create_dataset('488 Channel', data = Image_Intensity, compression = 'gzip')
      f.create_dataset('561 Channel', data = GUV_Images, compression = 'gzip')
'''