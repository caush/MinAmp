import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import typing

import math

class CadreExperimental:
    """
    Minimisation de l'amplitude d'une somme de signaux constitués d'une fondamentale et de ses harmoniques.
    Les harmoniques pourront être décalées selon des phases. Il s'agit de calculer lesdites phases afin de diminuer la différence entre le maximum et le minimum du signal.
    """

    def __init__(self, periode:float=1.0, nombrePhases:int=32, premierePhase=1, tailleSousEchantillon=512, nombreSousEchantillons=512, rondes=5, tailleBatch=64,
     device:str=None):  # device:typing.Literal["cpu", "cuda:0"]=None, exemplesParPhase=16, exemplesParParametre=512,
        """ 
        Paramètres
        ----------
        periode (float) : la période du signal. Va être multipliée par 2 * math.pi
        nombrePhases (int) : nombre de phases à considérer
        premierePhase (int) : première phase d'amplitude non nulle (1 signifie que toutes les phases sont considérées)
        exemplesParPhase (int) : nombre d'échantillons par phase pour espérer être présent dans tous les quadrants
        exemplesParParametre (int) : nombre d'exemples par paramètres (valeur à comparer avec le nombre de paramètres c.-à-d. de phases)
        tailleBatch (int) : nombre de lots de données traitées en parallèle sur le GPU
        device (str) : hardware ou doit être faite l'optimisation. Valeurs "cpu" ou None. Si None, le GPU sera utilisé si présent
        rondes (int) : nombre d'optimisations
        tailleSousEchantillon (int) : taille d'un sous échantillon
        nombreSousEchantillons (int) : nombre de sous échantillons
        
        Variables d'instance
        --------------------
        epsilon (float) : = periode / (tailleSousEchantillon * nombreSousEchantillons)
        amplitudeTest (int) : None dans le cas où il n'y a pas eu encore de tests
        rondesEffectives (int) : le nombre de rondes réslisées dans l'entrainement (0 si pas d'entrainement)
        """
        self.periode = periode
        self.nombrePhases = nombrePhases
        self.premierePhase = premierePhase
        self.tailleSousEchantillon = tailleSousEchantillon
        self.nombreSousEchantillons = nombreSousEchantillons
        self.tailleBatch = tailleBatch
        self.rondes = rondes

        self.epsilon = self.periode / (self.tailleSousEchantillon * self.nombreSousEchantillons)
        print(f"Le système est entrainé sur {self.nombreSousEchantillons} sous échantillons de taille {self.tailleSousEchantillon}.")
        print(f"Le ratio entre la taille de ces sous échantillons et le nombre de phases est de {tailleSousEchantillon / nombrePhases}.")
               
        self.training_data = CadreExperimental.RealDataset(self)
        self.test_data = CadreExperimental.RealDataset(self)

        # Create data loaders. Attention je me sers du batch_size pour créer l'échantillon. !!!
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.tailleSousEchantillon, shuffle = True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.tailleSousEchantillon)

        # Le device (cpu ou gpu) est calculé automatiquement sauf si on impose une valeur
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device==None else torch.device(device)
        print(f"Hardware utilisé : {self.device}")

        # Pour l'instant on ne peut pas changer cela d'une instance à l'autre
        self.model = CadreExperimental.NeuralNetwork(self).to(self.device)
        self.loss_fn = nn.L1Loss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        
        # On forme un identificateur avec les métaparamètres
        self.identificateur = f"{self.nombrePhases} phases (première {self.premierePhase}),"
        self.identificateur += f" {self.nombreSousEchantillons} sous échantillons de taille {self.tailleSousEchantillon} (periode {self.periode}),"
        self.identificateur += f" {self.rondes} rondes"

        # Pour l'instant le système n'a pas été testé, donc il n'y a pas d'amplitude pour les tests
        self.amplitudeTest = None
        self.rondesEffectives = 0

    # Un dataset de réels (très simple)
    class RealDataset(Dataset):
        def __init__(self, outer):
            self.min = 0.0
            self.max = outer.periode
            self.step = outer.epsilon

        def __len__(self):
            return int((self.max - self.min) / self.step)

        def __getitem__(self, idx):
            number = self.min + self.step * idx
            label = 0
            return number, label

    # Le modèle (très simple aussi)
    class NeuralNetwork(nn.Module):
        def __init__(self, outer):
            super(CadreExperimental.NeuralNetwork, self).__init__()
            
            # self.Amplitudes énumère les amplitudes nulles (celles du débuat, à commencer par la fondamentale)
            Amplitudes=torch.ones(outer.nombrePhases).float()
            Amplitudes[0:outer.premierePhase-1]=0
            self.Amplitudes=Amplitudes.reshape(outer.nombrePhases, 1).to(outer.device)

            # self.DeuxPiKSurT énumère les k de 1 (inclu) à nombrePhases (inclu) multipliés par les constantes 2, pi et 1/periode.
            # C'est un vecteur de float (pour aller dans la moulinette CUDA), colonne (c.-à-d. une matrice avec nombrePhases lignes et une colonne).
            DeuxPiKSurT = torch.arange(1, outer.nombrePhases+1).float().reshape(outer.nombrePhases, 1) * 2 * math.pi / outer.periode
            self.DeuxPiKSurT = DeuxPiKSurT.to(outer.device)

            # Les paramètres du modèle sont les phases (il y en a nombrePhases). 
            # On pourrait ramener à nombrePhases - 1 paramètre en considérant que la phase de la fondamentale est toujours 0. Mais on ne le fera pas.
            # self.phase = nn.Parameter(torch.zeros(outer.nombrePhases).reshape(outer.nombrePhases, 1))
            self.phase = nn.Parameter(torch.rand(outer.nombrePhases).reshape(outer.nombrePhases, 1) * 2 * math.pi) # Semble mieux marcher que des zéros partout

            # Il faut récupérer les variables de la classe externe (idiosyncrasie Python)
            self.tailleSousEchantillon  = outer.tailleSousEchantillon
            self.nombrePhases = outer.nombrePhases

        def forward(self, t:torch.tensor):
            # t représente un échantillon de taille tailleSousEchantillon des entrées
            # C'est un vecteur (dimension 1) qu'il faut transformer en vecteur ligne (dimension 2)
            t=t.reshape(1, self.tailleSousEchantillon)

            # On calcule une matrice de taille nombrePhases x tailleSousEchantillon
            kt = self.DeuxPiKSurT @ t
            
            # On additionne la phase et on prend le cosinus pour toutes les valeurs
            # Ensuite on supprime les valeurs pour les amplitudes nulles
            # Cela donne toujours une matrice de taille nombrePhases x tailleSousEchantillon
            lesCosinus = self.Amplitudes * torch.cos(kt + self.phase)

            # On fait la somme sur les phases, c'est à dire la première dimension (0)
            value = torch.sum(lesCosinus, 0) / self.nombrePhases #  math.sqrt(nombrePhases)
            #value_max = torch.max(lesCosinus, 0).values
            #value_min = torch.min(lesCosinus, 0).values
            return value

    # Boucle pour l'entrainement. On ne se sert pas des y !
    def train(self, dataloader, model, loss_fn, optimizer, trace):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            # Mettre des floats pour aller dans la moulinette CUDA
            X = X.float()
            # Évaluration du modèle : on récupère des points pour tout la courbe
            pred = model(X)

            # Calcul du min et du max
            pred_min = torch.min(pred)
            pred_max = torch.max(pred)
            
            # Minimisation de la valeur absolue entre le min et la max
            loss = loss_fn(pred_min, pred_max)
            
            # Propagation linéaire
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch % 256 == 0 and trace >=2):
                loss, current = loss.item(), batch * len(X)
                print(f"Amplitude(entrainement): {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Boucle pour les tests
    def test(self, dataloader, model):
        model.eval()
        test_min=1
        test_max=-1
        self.test_x= []
        self.test_y = []
        with torch.no_grad():
            for X, y in dataloader:
                X = X.float()
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                pred_min = torch.min(pred)
                pred_max = torch.max(pred)
                test_min = min(test_min, pred_min)
                test_max = max(test_max, pred_max)
                # On garde les valeurs pour le dessin éventuel
                self.test_x.append(X.cpu().numpy())
                self.test_y.append(pred.cpu().numpy())
        self.amplitudeTest = test_max - test_min # f" ({test_max - test_min:>8f})"

    # Pour entrainer un modèle défini
    def entraine(self, trace:int=2, dessine:int=1): 
        """
        Entrainement du système.

        Paramètres
        ----------
        trace (int) : 0 rien n'est tracé, 1 la derniere étape est tracée, 2 tout est tracé
        dessine (int) : 0 rien n'est dessiné, 1 la derniere étape est dessiné, 2 tout est dessiné
        """
        # On imprime le résultat sans entrainement pour les valeurs initiales
        if (dessine >= 2): self.Dessine()
        if (trace >=2): print(self.signature())
        for t in range(self.rondes):
            if (trace >=2): print(f"Ronde {t+1}")
            self.train(self.train_dataloader, self.model, self.loss_fn, self.optimizer, trace)
            self.rondesEffectives += 1
            self.test(self.test_dataloader, self.model)
            if (trace >=1): print(f"Amplitude(test): {self.amplitudeTest:>5f}")
            if ((dessine >= 1) and (t+1==self.rondes)): self.dessine()
 
    def parametres(self):
        """
        Retourne un tableau (numpy) de paramêtres.
        """
        return self.model.phase.cpu().detach().numpy().reshape(self.nombrePhases)

    def dessine(self):
        """
        Exécute un test et ensuite affiche le résultat
        """
        if self.amplitudeTest == None: self.test(self.test_dataloader, self.model)
        plt.figure(figsize=(20, 10))
        plt.title(self.signature())
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.plot(self.test_x, self.test_y, color = "black")
        plt.show()

    def sauve(self, repertoire="models"):
        """
        Sauve les paramètres du cadre expérimental dams un répertoire
        Le nom du fichier est constitué des métaparamètres et de l'amplification
        """
        import os
        filename=os.path.join(repertoire, self.signature() + ".pth")
        if self.amplitudeTest == None: self.test(self.test_dataloader, self.model)
        torch.save(self.model.state_dict(), filename)
        return self.signature()

    def sauveNew(self, filename="CadreExperimental.csv"):
        """
        Sauve les paramètres et l'amplification du cadre expérimental dans un fichier CSV 
        Il va y avoir une ligne titre pour le nom de ces paramètres
        En plus, il y a une colonne indiquant la date de la sauvegarde
        """
        import os
        import csv
        entetes = ["Phases", "Premiere", "Nombre SE", "Taille SE", "Periode", "Rondes", "Rondes E", "Amplitude", "Datetime", "Parametres"]
        # On crée un fichier avec les entêtes s'il n'existe pas
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                writer = csv.writer(f, lineterminator="\n")
                # writer=csv.writer(f, dialect="excel-tab")
                writer.writerow(entetes)
        
        # Renseigner l'amplitude
        if self.amplitudeTest == None:
            self.test(self.test_dataloader, self.model)
        
        # Forger la ligne
        from datetime import datetime
        ligne = [self.nombrePhases, self.premierePhase, self.nombreSousEchantillons, self.tailleSousEchantillon, self.periode, self.rondes, self.rondesEffectives, self.amplitudeTest.item(), datetime.today()]
        ligne.append(self.parametres().tolist())

        # Écrire la ligne à la fin du fichier
        with open(filename, "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            # writer=csv.writer(f, dialect="excel-tab")
            writer.writerow(ligne)

        return self.signature()

    def lire(self):
        """
        Lit les fichiers dont les noms sont consitués des métaparamêtres. Se concentre sur le fichier avec la meilleure amplitude (la plus faible).
        """
        import glob
        names=glob.glob(self.identificateur + "*.pth")
        names.sort()
        self.model.load_state_dict(torch.load(names[0]))
        self.test(self.test_dataloader, self.model)
        return self.signature()

    def lireNew(self, filename="CadreExperimental.csv"):
        """
        Lit les paramètres et l'amplitude. Se concentre sur le fichier avec la meilleure amplitude (la plus faible).
        """
        import pandas as pd

        import glob
        names=glob.glob(self.identificateur + "*.pth")
        names.sort()

        import ast
        fichier = pd.read_csv(filename)
        fichier[['Parametres']] = fichier[['Parametres']].applymap(ast.literal_eval)
 
        fichier=fichier[(fichier["Phases"]==self.nombrePhases) & 
                        (fichier["Premiere"]==self.premierePhase) & \
                        (fichier["Nombre SE"]==self.nombreSousEchantillons) & \
                        (fichier["Nombre SE"]==self.nombreSousEchantillons) & \
                        (fichier["Taille SE"]==self.tailleSousEchantillon) ]

        parametres = fichier.nsmallest(1, "Amplitude", keep='first')["Parametres"].values[0]

        self.model.phase[0] = parametres

        #self.model.load_state_dict(torch.load(names[0]))
        #self.test(self.test_dataloader, self.model)
        return self.signature()

    def signature(self):
        """
        Retourne la signature du cadre expérimental, c'est à dire les métaparamètres et l'amplitude
        """
        #if not hasattr(self, 'amplitudeTest'):
        if self.amplitudeTest == None:
            # obj.attr_name exists.
            self.test(self.test_dataloader, self.model)
        return self.identificateur +  f" ({self.amplitudeTest.item():>5f})"

if __name__ == '__main__':
    cadreExperimental = CadreExperimental()
    # params=cadreExperimental.entraine()
    cadreExperimental.sauve()
    cadreExperimental.lire()