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

#%% ESCALADO MULTIDIMENSIONAL 3 DIMENSIONES
#%%% Transformación

mds = MDS(n_components=3)
mds_result = mds.fit_transform(X_normalizado)

print("\nCoordenadas Proyectadas:")
print(mds_result)

# Imprime el stress
print(f"Stress: {mds.stress}")

#%%% Centrado de Coordenadas
mean_coords = np.mean(mds_result, axis=0)
mds_result_centered = mds_result - mean_coords
std_coords = np.std(mds_result_centered, axis=0)
mds_result_normalized = mds_result_centered / std_coords


componentes_principales_df = pd.DataFrame(mds_result_centered[:, :3], columns=['Componente 1', 'Componente 2', 'Componente 3'])

# Concatena el DataFrame de componentes principales con la Serie Y
resultado = pd.concat([Y, componentes_principales_df], axis=1)

#%%% Gráfica 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Lista de etiquetas de clase únicas
clases_unicas = resultado['class'].unique()

# Asigna un color distinto a cada clase
colores = ['b', 'g', 'r']

# Itera a través de cada clase y plotea los puntos con el color correspondiente
for i, clase in enumerate(clases_unicas):
    datos_clase = resultado[resultado['class'] == clase]
    ax.scatter(datos_clase['Componente 1'], datos_clase['Componente 2'], datos_clase['Componente 3'], c=colores[i], label=f'Clase {clase}')

# Etiquetas de los ejes
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')

# Leyenda
ax.legend()

# Mostrar la gráfica
plt.show()

#%%% clasificación

X_train, X_test, y_train, y_test = train_test_split(mds_result, Y, test_size=0.3, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_3 = accuracy_score(y_test, y_pred)
classification_report_3 = classification_report(y_test, y_pred)
confusion_3 = confusion_matrix(y_test, y_pred)

#%%% Métricas de evaluación

#   3 Dimensiones
print('_' * 55)  
print('\nResultados con 3 Dimensiones:')
print(f'\nPrecisión: {accuracy_3}')
print(f'\nInforme de clasificación:\n{classification_report_3}\n')

sns.heatmap(confusion_3, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión (3 dimensiones)')
plt.show()
