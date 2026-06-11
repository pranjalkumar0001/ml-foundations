import mne
from mne.io import read_raw_edf
from mne.io import concatenate_raws
Subject = 1
run = [4, 8, 12]
file_path = mne.datasets.eegbci.load_data(Subject, run)
raw_objects = []
for objects in file_path:
    data = read_raw_edf(objects, preload=True)
    raw_objects.append(data)
raw = concatenate_raws(raw_objects)
print(raw.info)