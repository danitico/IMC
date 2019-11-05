#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel Ranchal Parrado
"""

import pickle
import os
import click
import arff
import numpy as np
import random
import warnings
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss

warnings.filterwarnings('ignore')

@click.command()
@click.option('--train_file', '-t', default=None, required=True, type=str, show_default=True,
              help=u'Fichero con los datos de entrenamiento.')
@click.option('--test_file', '-T', default=None, required=False, type=str, show_default=True,
              help=u'Fichero con los datos de generalización.')
@click.option('--classification', '-c', required=False, type=bool, is_flag=True, show_default=True, default=False,
              help=u'Indica si el problema es de clasificación.')
@click.option('--ratio_rbf', '-r', default=0.1, required=False, type=float, show_default=True,
              help=u'Ratio de neuronas por patrón de entrenamiento.')
@click.option('--l2', '-l', required=False, type=bool, is_flag=True, show_default=True, default=False,
              help=u'Indica si se utiliza regularización L2')
@click.option('--eta', '-e', required=False, type=float, default=1e-2, show_default=True,
              help=u'Valor del parámetro eta')
@click.option('--outputs', '-o', required=False, type=int, default=1, show_default=True,
              help=u'Número de columnas de salidas que tiene el conjunto de datos')
@click.option('--model_file', '-m', default="", show_default=True,
              help=u'Fichero en el que se guardará o desde el que se cargará el modelo (si existe el flag p).')  # KAGGLE
@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Activar el modo de predicción.')  # KAGGLE
def entrenar_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file, pred):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Ejecución de 5 semillas.
    """
    if not pred:

        if train_file is None:
            print("No se ha especificado el conjunto de entrenamiento (-t)")
            return

        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)

        train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(train_file,
                                                                               test_file,
                                                                               outputs)

        for s in range(0, 5):
            print("-----------")
            print("Semilla: %d" % int(s+1))
            print("-----------")
            np.random.seed(s + 1)
            train_mses[s], test_mses[s], train_ccrs[s], test_ccrs[s] =\
                entrenar_rbf(train_inputs, train_outputs, test_inputs, test_outputs,
                             classification, ratio_rbf, l2, eta, outputs,
                             model_file and "{}/{}.pickle".format(model_file, s // 100) or "")

            print("MSE de entrenamiento: %f" % train_mses[s])
            print("MSE de test: %f" % test_mses[s])
            print("CCR de entrenamiento: %.2f%%" % train_ccrs[s])
            print("CCR de test: %.2f%%" % test_ccrs[s])

        print("*********************")
        print("Resumen de resultados")
        print("*********************")
        print("MSE de entrenamiento: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("MSE de test: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("CCR de entrenamiento: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("CCR de test: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))

    else:
        # KAGGLE
        if model_file is None:
            print("No se ha indicado un fichero que contenga el modelo (-m).")
            return

        # Obtener predicciones para el conjunto de test
        predictions = predict(test_file, model_file)

        # Imprimir las predicciones en formato csv
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""
            s += str(index)

            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))

            print(s)


def entrenar_rbf(train_inputs, train_outputs, test_inputs, test_outputs, classification, ratio_rbf, l2, eta, outputs,
                 model_file=""):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Una única ejecución.
        Recibe los siguientes parámetros:
            - train_inputs: Datos de los atributos de entrenamiento
            - train_outputs: Etiquetas de los datos de entrenamiento
            - test_inputs: Datos de los atributos de generalización
            - test_ouputs: Etiquetas de los datos de generalización
            - classification: True si el problema es de clasificacion.
            - ratio_rbf: Ratio (en tanto por uno) de neuronas RBF con 
              respecto al total de patrones.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1 (para regresión logística).
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento. 
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test. 
              En el caso de regresión, devolvemos un cero.
    """

    num_rbf = int(train_inputs.shape[0]*ratio_rbf)
    print("Número de RBFs utilizadas: %d" % (num_rbf))
    kmedias, distancias, centros = clustering(classification, train_inputs,
                                              train_outputs, num_rbf)

    radios = calcular_radios(centros, num_rbf)

    matriz_r = calcular_matriz_r(distancias, radios)

    if not classification:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)

    matriz_r_test = calcular_matriz_r(kmedias.transform(test_inputs), radios)


        # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification': classification,
            'radios': radios,
            'kmedias': kmedias
        }
        if not classification:
            save_obj['coeficientes'] = coeficientes
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)

    # # # # # # # # # # #

    if not classification:
        train_mse = mean_squared_error(np.matmul(matriz_r, coeficientes), train_outputs)
        test_mse = mean_squared_error(np.matmul(matriz_r_test, coeficientes), test_outputs)
        train_ccr = test_ccr = 0

    else:
        lb = OneHotEncoder()
        train_outputs_binarised = lb.fit_transform(train_outputs).toarray()
        test_outputs_binarised = lb.fit_transform(test_outputs).toarray()

        train_ccr = logreg.score(matriz_r, train_outputs)*100
        test_ccr = logreg.score(matriz_r_test, test_outputs)*100
        train_mse = mean_squared_error(train_outputs_binarised, logreg.predict_proba(matriz_r))
        test_mse = mean_squared_error(test_outputs_binarised, logreg.predict_proba(matriz_r_test))

    return train_mse, test_mse, train_ccr, test_ccr


def lectura_datos(fichero_train, fichero_test, outputs):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """

    datosarchivotrain = arff.load(open(fichero_train))
    datosarchivotest = arff.load(open(fichero_test))

    datatrain = np.array(datosarchivotrain['data'])
    datatest = np.array(datosarchivotest['data'])

    train_inputs = datatrain[:, 0:-outputs]
    train_outputs = datatrain[:, datatrain.shape[1]-1:datatrain.shape[1]+outputs]
    train_inputs = train_inputs.astype(np.float32)
    train_outputs = train_outputs.astype(np.float32)

    test_inputs = datatest[:, 0:-outputs]
    test_outputs = datatest[:, datatest.shape[1]-1:datatest.shape[1]+outputs]
    test_inputs = test_inputs.astype(np.float32)
    test_outputs = test_outputs.astype(np.float32)

    return train_inputs, train_outputs, test_inputs, test_outputs


def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir los patrones de forma estratificada, manteniendo
        la proporción de patrones por clase.
        Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """

    labels, percentage = np.unique(train_outputs, return_counts=True)
    percentage = percentage / train_inputs.shape[0]
    numCentroids = num_rbf * percentage

    if all(percentage == percentage[0]):
        numCentroids = numCentroids.astype(int)
        if numCentroids.sum() < num_rbf:
            for _ in range(num_rbf - numCentroids.sum()):
                numCentroids[random.randint(0, numCentroids.shape[0]-1)] += 1
    else:
        numCentroids = numCentroids.round(0)
        numCentroids = numCentroids.astype(int)
        if numCentroids.sum() > num_rbf:
            for _ in range(numCentroids.sum() - num_rbf):
                numCentroids[random.randint(0, numCentroids.shape[0] - 1)] -= 1

    centroides = np.empty(0)

    for i, j in zip(labels, range(len(labels))):
        aux = np.where(train_outputs == i)[0]
        indices = random.sample(list(aux), int(numCentroids[j]))
        chosen = np.take(train_inputs, indices, axis=0)
        if j==0:
            centroides = np.copy(chosen)
        else:
            centroides = np.append(centroides, chosen, axis=0)

    return centroides


def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """

    if clasificacion:
        centros_iniciales = inicializar_centroides_clas(train_inputs, train_outputs, num_rbf)
    else:
        centros_iniciales = np.take(train_inputs, random.sample(range(len(train_inputs)), num_rbf), axis=0)

    kmedias = KMeans(n_clusters=num_rbf, init=centros_iniciales, n_init=1, max_iter=500, n_jobs=-1)
    kmedias.fit(train_inputs)

    centros = kmedias.cluster_centers_

    distancias = kmedias.transform(train_inputs)

    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """
    radios = np.zeros(num_rbf)
    for i in range(num_rbf):
        for j in range(num_rbf):
            if i != j:
                radios[i] += spatial.distance.euclidean(centros[i], centros[j])

    radios = radios / (2*(num_rbf-1))

    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """
    matriz_r = np.zeros(shape=(distancias.shape[0], distancias.shape[1] + 1))
    matriz_r[:, matriz_r.shape[1]-1] = 1
    matriz_r[:, 0:matriz_r.shape[1]-1] = np.copy(np.power(distancias, 2))

    for i in range(distancias.shape[1]):
        matriz_r[:, i] /= (-2*np.power(radios[i], 2))

    matriz_r[:, 0:matriz_r.shape[1] - 1] = np.exp(matriz_r[:, 0:matriz_r.shape[1] - 1])

    return matriz_r


def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """

    if matriz_r.shape[0] == matriz_r.shape[1] - 1:
        inversa = np.linalg.inv(matriz_r)
    else:
        inversa = np.linalg.pinv(matriz_r)

    coeficientes = np.matmul(inversa, train_outputs)

    return coeficientes


def logreg_clasificacion(matriz_r, train_outputs: np.ndarray, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    if l2:
        logreg = LogisticRegression(solver='liblinear', C=1/eta, n_jobs=-1)
    else:
        logreg = LogisticRegression(penalty='l1', solver='liblinear', C=1/eta, n_jobs=-1)

    logreg.fit(matriz_r, train_outputs)

    return logreg


def predict(test_file, model_file):
    """ Calcula las predicciones para un conjunto de test que recibe como parámetro. Para ello, utiliza un fichero que
    contiene un modelo guardado.
    :param test_file: fichero csv (separado por comas) que contiene los datos de test.
    :param model_file: fichero de pickle que contiene el modelo guardado.
    :return: las predicciones para la variable de salida del conjunto de datos proporcionado.
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]

    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)

    radios = saved_data['radios']
    classification = saved_data['classification']
    kmedias = saved_data['kmedias']

    test_distancias = kmedias.transform(test_inputs)
    test_r = calcular_matriz_r(test_distancias, radios)

    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coeficientes']
        test_predictions = np.dot(test_r, coeficientes)

    return test_predictions


if __name__ == "__main__":
    entrenar_rbf_total()
