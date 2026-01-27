import nibabel as nib
import numpy as np
g = nib.load("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
data = g.darrays[0].data
print("Unique IDs in Left Atlas:", np.unique(data))

r = nib.load("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
datar = r.darrays[0].data
print("Unique IDs in Right Atlas:", np.unique(datar))