import mne
from mne.io import read_raw_edf
from mne.io import concatenate_raws
import matplotlib.pyplot as plt
Subject = 1
run = [4, 8, 12]
file_path = mne.datasets.eegbci.load_data(Subject, run)
raw_objects = []
for objects in file_path:
    data = read_raw_edf(objects, preload=True)
    raw_objects.append(data)
raw = concatenate_raws(raw_objects)
#print(raw.info)
#raw.compute_psd().plot() # plot without filter
# plt.show()
raw.filter(l_freq=8, h_freq= 30)
#raw.compute_psd().plot()
raw.notch_filter(freqs=50)
#raw.compute_psd().plot()
#plt.show()
events, event_id_dict = mne.events_from_annotations(raw)
print("printing event_id_dict")
print(event_id_dict) #T00: rest, T1: left hand, T2: right hand
custom_mapping = {'rest': 1, 'left_fist': 2, 'right_fist': 3}
epochs = mne.Epochs(raw,events, event_id=custom_mapping, tmin= -1, tmax= 4, preload=True)
print(epochs)
