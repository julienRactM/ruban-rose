from PIL import Image
import numpy as np

path=input("quel est le nom de l'image à amplifier ? ")

image = Image.open(path).convert("RGB")

#conversion en numpy array pour travailler les valeur
img_array = np.array(image).astype(np.int16)

moyenne = np.mean(img_array, axis=(0, 1), keepdims=True)


#on emplifie les différence
diff = img_array - moyenne
amplified = moyenne + diff * 20


#on évite les valeurs invalide
amplifié = np.clip(amplified, 0, 255).astype(np.uint8)


#on sauvegarde l'image
Image.fromarray(amplifié).save(input("quel est le nom de l'image de sortie ? "))
