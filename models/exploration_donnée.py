import zipfile
from PIL import Image
from io import BytesIO
import numpy as np

path = 'BHI.zip'

ZIP=zipfile.ZipFile(path, 'r')

filename = ZIP.namelist()

compteur = {'0': 0, '1': 0}

for name in filename :
    if name.split('/')[1]=='0' :
        compteur['0']=compteur['0']+1
    if name.split('/')[1]=='1' :
        compteur['1']=compteur['1']+1

print(compteur)
input()

#on récupère le nombre total d'image
n=sum(compteur.values())

#récupère les dimensions de la première image
file=ZIP.open(name)
pic=BytesIO(file.read())
picture=Image.open(pic)
largeur, hauteur = picture.size

#Initialisation des accumulateurs
somme = np.zeros((hauteur, largeur, 3), dtype=np.float64)
somme_carre = np.zeros((hauteur, largeur, 3), dtype=np.float64)

#calcule de la moyenne
for name in filename :
    file=ZIP.open(name)
    pic=BytesIO(file.read())
    picture=Image.open(pic)
    for y in range(hauteur):
        for x in range(largeur):
            try :
                r, g, b = picture.getpixel((x, y))
                somme[y, x, 0] += r
                somme[y, x, 1] += g
                somme[y, x, 2] += b
            except :
                print("pas la même taille,"+str(x)+','+str(y)+','+name)
                break

moyenne = (somme / n).astype(np.uint8)

img_moyenne = Image.fromarray(moyenne)

img_moyenne.show()
img_moyenne.save("moy.png", format="PNG")


for name in filename :
    file=ZIP.open(name)
    pic=BytesIO(file.read())
    picture=Image.open(pic)
    for y in range(hauteur):
        for x in range(largeur):
            try :
                r, g, b = picture.getpixel((x, y))
                somme_carre[y, x, 0] += (r-moyenne[y, x, 0])**2
                somme_carre[y, x, 1] += (g-moyenne[y, x, 1])**2
                somme_carre[y, x, 2] += (b-moyenne[y, x, 2])**2
            except :
                print("pas la même taille,"+str(x)+','+str(y)+','+name)
                break

ecart_type = (np.sqrt(somme_carre/n)).astype(np.uint8)
img_ecart_type = Image.fromarray(ecart_type)

img_ecart_type.save("écart_type.png", format="PNG")
