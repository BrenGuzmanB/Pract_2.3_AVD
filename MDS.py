"""
Created on Fri Oct 27 22:31:09 2023

@author: Bren Guzmán
"""
#       MULTIDIMENSIONAL SCALING


import numpy as np

class MDS:
    def __init__(self, n_components):
        self.n_components = n_components
        self.stress = None

    def fit_transform(self, data):
        # Paso 1: Calcular la matriz de distancias
        distance_matrix = self.euclidean_distance_matrix(data)

        # Paso 2: Aplicar centrado simple
        n = distance_matrix.shape[0]
        dist_sq = distance_matrix ** 2
        H = np.eye(n) - np.ones((n, n)) / n

        B = -0.5 * np.dot(np.dot(H, dist_sq), H)

        # Paso 3: Resolver los valores propios y los vectores propios
        eigenvalues, eigenvectors = np.linalg.eigh(B)

        # Paso 4: Resolver las coordenadas proyectadas
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = np.dot(eigenvectors[:, :self.n_components], np.diag(np.sqrt(eigenvalues[:self.n_components])))

        # Calcular el stress
        self.stress = self.calculate_stress(distance_matrix, self.components_)

        return self.components_

    def euclidean_distance_matrix(self, data):
        n = data.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(data.loc[i] - data.loc[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def calculate_stress(self, original_distance_matrix, projected_data):
        n = original_distance_matrix.shape[0]
        projected_distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(projected_data[i] - projected_data[j])
                projected_distance_matrix[i, j] = dist
                projected_distance_matrix[j, i] = dist

        stress = np.sum((original_distance_matrix - projected_distance_matrix) ** 2)
        return stress

