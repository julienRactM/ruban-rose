import zipfile
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'BHI.zip'

ZIP=zipfile.ZipFile(path, 'r')

filename = ZIP.namelist()

n=0

hist_r_list=[]
hist_g_list=[]
hist_b_list=[]
cat=input("tapez 1 ou 0 ")
for name in filename :
    if name.split('/')[1]==cat :
        file=ZIP.open(name)
        pic=BytesIO(file.read())
        picture=Image.open(pic).convert("RGB")
        img_rgb= np.array(picture)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
        colors = ('r', 'g', 'b')
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        hist_r_list.append(hist_r/hist_r.sum()*100)
        hist_g_list.append(hist_g/hist_r.sum()*100)
        hist_b_list.append(hist_b/hist_r.sum()*100)
        n=n+1

# Moyenne des histogrammes
hist_r_avg = np.mean(hist_r_list, axis=0)
hist_g_avg = np.mean(hist_g_list, axis=0)
hist_b_avg = np.mean(hist_b_list, axis=0)

# Affichage
plt.figure(figsize=(8, 4))
plt.plot(hist_r_avg, color='r', label='Rouge')
plt.plot(hist_g_avg, color='g', label='Vert')
plt.plot(hist_b_avg, color='b', label='Bleu')
plt.title("Histogramme moyen en pourcentage (RGB)")
plt.xlabel("Intensité")
plt.ylabel("Pourcentage de pixels (%)")
plt.legend()
plt.grid(True)
plt.xlim([0, 256])
plt.show()

#écart-type
hist_r_std = np.std(hist_r_list, axis=0)
hist_g_std = np.std(hist_g_list, axis=0)
hist_b_std = np.std(hist_b_list, axis=0)

# Affichage
plt.figure(figsize=(8, 4))
plt.plot(hist_r_std, color='r', label='Rouge')
plt.plot(hist_g_std, color='g', label='Vert')
plt.plot(hist_b_std, color='b', label='Bleu')
plt.title("Histogramme écart-type en pourcentage (RGB)")
plt.xlabel("Intensité")
plt.ylabel("Pourcentage de pixels (%)")
plt.legend()
plt.grid(True)
plt.xlim([0, 256])
plt.show()