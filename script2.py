# code pour montrer la survie en fonction du prix...
# mais il y a une petite errreur sur cette ligne.... toute fois en exécutant le code,
# on obtient bien le graphe montrant la survie en fonction du prix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# je récupère les données


titanic_train_data = pd.read_csv('data/train.csv', sep=',', header=0)
# import du fichier test qui servira à tester si une personne entre ou pris au hassard a survecu ou non
titanic_test_data = pd.read_csv('data/test.csv', sep=',', header=0)


titanic_train_data.head(n=7)


# je les étudies (graphs, statistiques, signification, ...) c'est à dire pour afficher

plt.hist(titanic_train_data['Sex'], bins=2)
plt.show()


# ....nombre de personne decede en fonction de leur age


# tracer la survie avec le prix


plt.figure(figsize=(20, 8))


survived_fare = titanic_train_data[titanic_train_data['Survived'] == 1]['Fare']
dead_fare = titanic_train_data[titanic_train_data['Survived'] == 0]['Fare']


plt.hist([survived_fare, dead_fare], bins=50, stacked=True,
         rwidth=0.8, color=['green', 'red'], label=['Survived', 'Dead'])

plt.xlabel('Tarif')
plt.ylabel('Nombre de passagers')

plt.axes([0, 550, 0, 350])
plt.grid()
# je sais pas ce que tu as voulu faire avec le legend :
# plt.legend(True)
plt.show()


# cela montre que les personnes qui avaient réservé des billets à bas prix sont les plus décédées
