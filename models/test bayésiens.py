import numpy as np
import zipfile
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import sklearn.model_selection as skm
import sklearn.metrics as metric
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class NaiveBayes:
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0) + 1e-9  # éviter division par zéro
            self.priors[cls] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[cls]))
            likelihood -= 0.5 * np.sum(((x - self.mean[cls]) ** 2) / self.var[cls])
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

path = 'BHI.zip'

ZIP=zipfile.ZipFile(path, 'r')

filename = ZIP.namelist()

Data=[]
cat=[]
for name in filename :
    file=ZIP.open(name)
    pic=BytesIO(file.read())
    picture=Image.open(pic).convert("RGB")
    img_rgb= np.array(picture)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    colors = ('r', 'g', 'b')
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
    Data.append(np.concatenate([hist_r/hist_r.sum()*100,hist_g/hist_r.sum()*100, hist_b/hist_r.sum()*100], axis=0).flatten())
    if name.split('/')[1]=='0' :
        cat.append('sain')
    else :
        cat.append('cancer')

X_train, X_test, y_train, y_test=skm.train_test_split(np.array(Data),np.array(cat),test_size=0.3)

N=NaiveBayes()
N.fit(X_train,y_train)
val=N.predict(X_test)

print(metric.accuracy_score(y_test,val))

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metric.accuracy_score(y_pred,val))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print(metric.accuracy_score(y_pred_knn,val))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64),activation='relu',solver='adam',max_iter=500,random_state=42)

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print(metric.accuracy_score(y_pred,val))



