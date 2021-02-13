import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# je récupère les données



titanic_train_data = pd.read_csv('data/train.csv',sep=',',header = 0)
titanic_test_data =pd.read_csv('data/test.csv',sep=',',header = 0) # import du fichier test qui servira à tester si une personne entre ou pris au hassard a survecu ou non


titanic_train_data.head(n=7)



# je les étudies (graphs, statistiques, signification, ...) c'est à dire pour afficher

  # montre clairement que les femmes ont survécu plus que les hommes

plt.hist(titanic_train_data['Sex'], bins=2) # nombre de personnes decede en fonction du sex
plt.show()

#....nombre de personne decede en fonction de la classe qu'elle occupait sur le bateau

plt.hist(titanic_train_data['Pclass'], bins=2)
plt.show()




# je nettoie et séléctionne celles qui me semble les plus pertinentes
embarked_to_number = {'C': 1, 'Q': 2,'S' : 3 }
titanic_train_data['Embarked_to_number'] = titanic_train_data['Embarked'].map(embarked_to_number)

sex_to_number={'male':1,'female':2}
titanic_train_data['Sex_to_number'] = titanic_train_data['Sex'].map(sex_to_number)


titanic_parameters = ['Fare','Age','Sex_to_number','Embarked_to_number','Pclass','Survived']
titanic_data_selected_to_clean = titanic_train_data[titanic_parameters]
titanic_data_cleaned_no_nan = titanic_data_selected_to_clean.dropna()


titanic_parameters = ['Fare','Age','Sex_to_number','Embarked_to_number','Pclass']
titanic_train_data_selected_no_nan = titanic_data_cleaned_no_nan[titanic_parameters]
titanic_train_data_survived = titanic_data_cleaned_no_nan['Survived']



# je créer mon modèle
from sklearn.linear_model import LogisticRegression

sklearn_LogisticRegression = LogisticRegression(max_iter=100)


sklearn_SVC = SVC(kernel = 'linear', C=1)

# J'apprend sur les données
sklearn_SVC.fit(titanic_train_data_selected_no_nan,titanic_train_data_survived)

sklearn_LogisticRegression.fit(titanic_train_data_selected_no_nan, titanic_train_data_survived)


# Je prédis
predicted_titanic_train_data_survived = sklearn_SVC.predict(titanic_train_data_selected_no_nan)
predicted_titanic_train_data_survived_logistic = sklearn_LogisticRegression.predict(titanic_train_data_selected_no_nan)


# Je mesure la qualité de ma prédiction
print("Resultats SVC")

print("\n")
print("Resultats régression logistique")
from sklearn.metrics import classification_report

print(classification_report(titanic_train_data_survived, predicted_titanic_train_data_survived))

print("----------------------------------------------------------------------")

##### utilisation d'un autre modele pour prédire

# prediction en utilisant une classification aléatoire
from sklearn.ensemble import RandomForestClassifier

y = titanic_train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(titanic_train_data[features])
X_test = pd.get_dummies(titanic_test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId' : titanic_test_data.PassengerId , 'Survived': predictions})
output = pd.DataFrame({'Name' : titanic_test_data.Name , 'Survived': predictions})

print("Resultats classification aléatoire")

print(output)
