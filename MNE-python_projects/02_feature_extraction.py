import mne

epoches = mne.read_epochs('subject1-epo.fif')
X = epoches.get_data()
y = epoches.events[:,-1]
print(X.shape, "\t", y.shape)