"""
Created on Sat Oct 28 18:12:15 2023

@author: Brenda Guzmán, Brenda García, María José Merino
"""
#%% LIBRERÍAS

import pandas as pd
import numpy as np
from MDS import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

#%% CARGAR ARCHIVO

columns_names = ["class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "0D280_0D315_of_diluted_wines", "Proline"]
df = pd.read_csv("Wine.csv", names= columns_names )

#%% SEPARAR X y Y

X = df.drop(columns=["class"])
Y = df["class"]

media_por_columna = np.mean(X, axis=0)
X_normalizado = X - media_por_columna

desviacion_estandar = np.std(X_normalizado, axis=0)
X_normalizado = X_normalizado / desviacion_estandar
#%% REGRESIÓN LOGÍSTICA (DATOS ORIGINALES)
#%%% Clasificación

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_og = accuracy_score(y_test, y_pred)
classification_report_og = classification_report(y_test, y_pred)
confusion_og = confusion_matrix(y_test, y_pred)

#%%% Métricas de evaluación

#   Datos originales
print('_' * 55)  
print('\nResultados con los datos originales:')
print(f'\nPrecisión: {accuracy_og}')
print(f'\nInforme de clasificación:\n{classification_report_og}\n')

sns.heatmap(confusion_og, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión (Datos originales)')
plt.show()
plt.show()