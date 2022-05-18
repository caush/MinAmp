import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import typing

import math

class CadreExperimetal:
    """
    Minimisation de l'amplitude d'une somme de signaux constitués d'une fondamentale et de ses harmoniques.
    Les harmoniques pourront être décalées selon des phases. Il s'agit de calculer lesdites phases afin de diminuer la différence entre le maximum et le minimum du signal.
    """

    def __init__(self, periode:float=1.0, nombrePhases:int=32, premierePhase=1, exemplesParPhase=16, exemplesParParametre=512, tailleBatch=64,
     device:str=None, epochs=5):  # device:typing.Literal["cpu", "cuda:0"]=None,
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
        epochs (int) : nombre d'optimisations

        Variables d'instance
        --------------------
        tailleEchantillon (int) : = nombrePhases * exemplesParPhase
        nombreEntrees (int) : =  tailleEchantillon * exemplesParParametre
        epsilon (float) : = periode / nombreEntrees

        """
        self.periode = periode
        self.nombrePhases = nombrePhases
        self.premierePhase = premierePhase
        self.tailleBatch = tailleBatch
        self.epochs = epochs

        self.tailleEchantillon = self.nombrePhases * exemplesParPhase
        self.nombreEntrees = self.tailleEchantillon * exemplesParParametre
        self.epsilon = self.periode / self.nombreEntrees
       
        self.training_data = CadreExperimetal.RealDataset(self)
        self.test_data = CadreExperimetal.RealDataset(self)

        # Create data loaders. Attention je me sers du batch_size pour créer l'échantillon. !!!
        self.train_dataloader = DataLoader(self.training_data, batch_size=self.tailleEchantillon, shuffle = True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.tailleEchantillon)

        # Le device (cpu ou gpu) est calculé automatiquement sauf si on impose une valeur
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device==None else torch.device(device)
        print(f"Hardware utilisé : {self.device}")

        # Pour l'instant on ne peut pas changer cela d'une instance à l'autre
        self.model = CadreExperimetal.NeuralNetwork(self).to(self.device)
        self.loss_fn = nn.L1Loss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        
        # On forme un identificateur avec les métaparamètres
        self.identificateur = f"{self.nombrePhases} phases (première {self.premierePhase}),"
        self.identificateur += f" lots de {self.tailleBatch}, échantillon de {self.tailleEchantillon},"
        self.identificateur += f" {self.nombreEntrees} entrées, {self.epochs} rondes"

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
            super(CadreExperimetal.NeuralNetwork, self).__init__()
            
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
            self.tailleEchantillon  = outer.tailleEchantillon
            self.nombrePhases = outer.nombrePhases

        def forward(self, t:torch.tensor):
            # t représente un échantillon de taille tailleEchantillon des entrées
            # C'est un vecteur (dimension 1) qu'il faut transformer en vecteur ligne (dimension 2)
            t=t.reshape(1, self.tailleEchantillon)

            # On calcule une matrice de taille nombrePhases x tailleEchantillon
            kt = self.DeuxPiKSurT @ t
            
            # On additionne la phase et on prend le cosinus pour toutes les valeurs
            # Ensuite on supprime les valeurs pour les amplitudes nulles
            # Cela donne toujours une matrice de taille nombrePhases x tailleEchantillon
            lesCosinus = self.Amplitudes * torch.cos(kt + self.phase)

            # On fait la somme sur les phases, c'est à dire la première dimension (0)
            value = torch.sum(lesCosinus, 0) / self.nombrePhases #  math.sqrt(nombrePhases)
            #value_max = torch.max(lesCosinus, 0).values
            #value_min = torch.min(lesCosinus, 0).values
            return value

    # Boucle pour l'entrainement. On ne se sert pas des y !
    def train(self, dataloader, model, loss_fn, optimizer):
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

            if batch % 256 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
        self.amplitudeTest = f" ({test_max - test_min:>8f})"
        
        return model.phase.cpu().detach().numpy()

    # Pour entrainer un modèle défini
    def entraine(self, matplot:str="dernier"):  # , matplot:typing.Literal["tout", "rien", "dernier", "premier"]="dernier"
        # On imprime le résultat sans entrainement pour les valeurs initiales
        self.test(self.test_dataloader, self.model) # matplot=plt.isinteractive()
        print(self.identificateur + self.amplitudeTest)
        if (matplot=="tout" or matplot=="premier"):
            self.matplot()

        for t in range(self.epochs):
            print(f"\nRonde {t+1}") # \n-------------------------------
            self.train(self.train_dataloader, self.model, self.loss_fn, self.optimizer)

            params = self.test(self.test_dataloader, self.model)
            print(self.amplitudeTest)
            if (matplot=="tout" or (matplot=="dernier" and (t+1==self.epochs))):
                self.matplot()
            
        return params.reshape(self.nombrePhases)

    # Pour faire un graphique de la dernière solution testée
    def matplot(self):
        plt.figure(figsize=(20, 10))
        plt.title(self.identificateur + self.amplitudeTest)
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.plot(self.test_x, self.test_y, color = "black")
        plt.show()

    def sauver(self):
        torch.save(self.model.state_dict(), self.identificateur + self.amplitudeTest + ".pth")

    def lire(self):
        import glob
        name=glob.glob(self.identificateur + "*.pth")[0]
        self.model.load_state_dict(torch.load(name))
        self.test(self.test_dataloader, self.model)
        return self.amplitudeTest


if __name__ == '__main__':
    cadreExperimental = CadreExperimetal()
    params=cadreExperimental.entraine()
    print("\nPhases\n", params)