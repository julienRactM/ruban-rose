import zipfile
from PIL import Image
from io import BytesIO
import numpy as np
import random as rnd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sklearn.metrics as smt
import numpy as np
import optuna


def split(List,per) :
    test=[]
    while len(test)<len(List)*per/100 :
        n=rnd.randint(0,len(List)-1)
        test.append(List[n])
        List.remove(List[n])
    return test,List


def batch(ZIP,clas1,clas2) :
    batch1=[]
    for name in clas1 :
        file=ZIP.open(name)
        pic=BytesIO(file.read())
        try :
            picture=Image.open(pic)
            batch1.append(picture)
        except Exception as e:
            print(f"Image qui ne marche pas : {name}")
            print(f"Erreur : {e}")

    batch2=[]
    for name in clas2 :
        file=ZIP.open(name)
        pic=BytesIO(file.read())
        try :
            picture=Image.open(pic)
            batch2.append(picture)
        except Exception as e:
            print(f"Image qui ne marche pas : {name}")
            print(f"Erreur : {e}")
    return batch1,batch2

def fit(model,X,Y,optimizer) :  # classe 0 et 1
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    print(model.train_on_batch(X, Y, criterion, optimizer))
    return model


def batch_learn(zip,model,optimizer,clas1,clas2,batch_size,learnstep,imsize) :
    batch1=[]
    batch2=[]
    start=0
    while start<max(len(clas1),len(clas2)) :
        if len(clas1)>start :
            if len(clas1)<start+batch_size :
                batch1=clas1[start:len(clas1)-1]
            else :
                batch1=clas1[start:start+batch_size]
        if len(clas2)>start :
            if len(clas2)<start+batch_size :
                batch2=clas2[start:len(clas2)-1]
            else :
                batch2=clas2[start:start+batch_size]
        print(len(batch1),len(batch2))
        Batch1,Batch2=batch(zip,batch1,batch2)
        if len(Batch1)>0 and len(Batch2)>0 :
            while len(Batch1)>len(Batch2) :
                angle = rnd.choice([0, 90, 180, 270])
                Batch2.append(Batch2[-1].rotate(-angle))
            while len(Batch1)<len(Batch2) :
                angle = rnd.choice([0, 90, 180, 270])
                Batch1.append(Batch1[-1].rotate(-angle))
            Batch=Batch1+Batch2
            for im in Batch :
                im.resize(imsize)
            Y=[0]*len(Batch1)+[1]*len(Batch2)
            for i in range(learnstep) :
                model=fit(model,Batch,Y,optimizer)
        start=start+batch_size
        print(start)
        print(max(len(clas1),len(clas2)))
    return model

path = 'test.zip'

ZIP=zipfile.ZipFile(path, 'r')

filename = ZIP.namelist()

sain=[]
cancer=[]

for name in filename :
    if name.split('/')[1]=='0' :
        sain.append(name)
    if name.split('/')[1]=='1' :
        cancer.append(name)



class SimpleViT(nn.Module):
    def __init__(self,embed_dim=3, num_heads=4, num_layers=2):
        super().__init__()
        num_classes=2
        # Token CLS (classe)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer Encoder (PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Tête de classification
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: Tensor de shape (batch_size, num_patches, embed_dim)
        """
        try :
            E=[]
            for e in range(len(x)) :
                E.append(list(x[e].getdata()))
            x=E
            print(x)
            x=torch.tensor(x, dtype=torch.float32)

            B, N, D = x.shape

            # Répéter token CLS pour batch
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)

            # Concat CLS token devant les patches
            x = torch.cat((cls_tokens, x), dim=1)           # (B, 1 + num_patches, embed_dim)

            # créer les embeddings positionnels à la volée
            pos_embed = torch.randn(1, x.size(1), D, device=x.device)

            print(f"x.shape: {x.shape}")
            print(f"pos_embed.shape: {pos_embed.shape}")

            # Ajouter embeddings positionnels
            x = x + pos_embed

            # Transformer attend input shape (sequence_len, batch, embed_dim)
            x = x.permute(1, 0, 2)
            
            # Passage dans le Transformer Encoder
            x = self.transformer(x)

            # Récupérer la sortie du token CLS (premier token)
            x = x[0]

            # Classification
            x = self.mlp_head(x)
            return x
        except :
            print("Échec dans forward:", str(e))
            return torch.zeros(1, self.mlp_head.out_features)

    
    def train_on_batch(self, batch_x, batch_y, criterion, optimizer):
        print("batch x : ",len(batch_x),"batch y :",len(batch_y))
        print(batch_y)
        try :
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(batch_x)
            batch_y = torch.tensor(batch_y, dtype=torch.long)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            return loss.item()
        except :
            print("invalid batch")


def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [3,6,9,12])
    num_layers = trial.suggest_int("num_layers", 3,10)
    batch_size = trial.suggest_int("batch_size", 1,3)
    learnstep = trial.suggest_int("batch_size", 1,6)
    imsize = trial.suggest_categorical("imsize", [(50,50),(70,70),(100,100),(150,150),(200,200)])
    
    
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    model= SimpleViT(embed_dim=embed_dim,num_heads=3,num_layers=num_layers)
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    path = 'test.zip'
    
    ZIP=zipfile.ZipFile(path, 'r')

    filename = ZIP.namelist()
    
    sain=[]
    cancer=[]

    for name in filename :
        if name.split('/')[1]=='0' :
            sain.append(name)
        if name.split('/')[1]=='1' :
            cancer.append(name)

    
    trainsain,testsain=split(sain,70)

    traincancer,testcancer=split(cancer,70)
    
    # Charger les données et entraîner
    trained_model = batch_learn(ZIP,model,optimizer,trainsain,traincancer,batch_size=batch_size,learnstep=learnstep,imsize=imsize)

    # Évaluation : ici, on utilise le modèle sur les données de test
    y_pred = []
    y_true = []

    for name in testsain:
        try:
            file = ZIP.open(name)
            pic = BytesIO(file.read())
            picture = Image.open(pic)
            out = trained_model.forward([picture])
            y_pred.append(torch.argmax(out).item())
            y_true.append(0)
        except:
            continue

    for name in testcancer:
        try:
            file = ZIP.open(name)
            pic = BytesIO(file.read())
            picture = Image.open(pic)
            out = trained_model.forward([picture])
            y_pred.append(torch.argmax(out).item())
            y_true.append(1)
        except:
            continue

    # Calcul du recall
    if len(y_true) == 0 or len(set(y_pred)) < 2:
        return 0.0  # cas extrême, évite erreur

    recall = smt.recall_score(y_true, y_pred, average="macro")
    return recall

study = optuna.create_study(direction="maximize")  # maximize car tu veux un haut recall
study.optimize(objective, n_trials=20)  # tu peux augmenter n_trials ensuite

# Afficher les meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres :", study.best_params)
