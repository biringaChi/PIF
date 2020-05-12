__author__ = 'biringaChi'
__email__ = "biringachidera@gmail.com"

from matplotlib import pyplot as plt
import pickle

G_losses6 = open("metrics/G_losses6.pickle","rb")
G_losses6 = pickle.load(G_losses6)

G_losses6_2 = open("metrics/G_losses6_2.pickle","rb")
G_losses6_2 = pickle.load(G_losses6_2)

combined = []

for loss in G_losses6:
    combined.append(loss)

for loss_2 in G_losses6_2:
    combined.append(loss_2)


def network_losses(combined):
    plt.figure(figsize=(10,5))
    plt.title("Loss Rate")
    plt.plot(combined)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.show()

network_losses(combined)

