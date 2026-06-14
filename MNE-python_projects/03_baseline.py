import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

epoches = mne.read_epochs("subject1-epo.fif")
X = epoches.get_data()
y = epoches.events[:,-1]
X_rest = []
X_left = []
X_right = []
indices = np.where((y==2) | (y==3))[0]
X_binary = X[indices]
y_binary = y[indices]
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()
clf = Pipeline([('CSP', csp), ('LDA', lda)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_binary, y_binary, cv=cv)

print(f"Fold Scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores)}")

np.save("X_binary.npy", X_binary)
np.save("y_binary.npy", y_binary)