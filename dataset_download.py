import os

os.environ['KAGGLEHUB_CACHE'] = '/workspace'

import kagglehub

path = kagglehub.dataset_download("washingtongold/lidcidri30")

print("Downloaded to:", path)

import glob

dicom_files = glob.glob(os.path.join(path, "**/*.dcm"), recursive=True)

print(f"Found {len(dicom_files)} DICOM files")

print("Sample path:", dicom_files[0])

import pydicom

ds = pydicom.dcmread(dicom_files[0])

print(ds)  # shows all metadata

import numpy as np

image = ds.pixel_array

print("Shape:", image.shape)

print("Dtype:", image.dtype)