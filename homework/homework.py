# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
 
# flake8: noqa: E501
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import gzip
import pickle
import pandas as pd
import json
import os


def procesar_datos(df):
    data = df.copy()
    data.drop(columns='ID', inplace=True)
    data.rename(columns={'default payment next month': 'default'}, inplace=True)
    data.dropna(inplace=True)
    data = data[(data['EDUCATION'] != 0) & (data['MARRIAGE'] != 0)]
    data.loc[data['EDUCATION'] > 4, 'EDUCATION'] = 4
    return data


def construir_pipeline():
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_cols = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('numerical', MinMaxScaler(), num_cols)
        ],
        remainder='passthrough'
    )

    selector = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', selector),
        ('logreg', LogisticRegression(max_iter=1000, solver='saga', random_state=42))
    ])
    return pipeline


def ajustar_modelo(pipeline, cv_folds, X_train, y_train, metric):
    params = {
        'feature_selection__k': range(1, 11),
        'logreg__penalty': ['l1', 'l2'],
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=cv_folds,
        scoring=metric,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid


def obtener_metricas(modelo, X_train, y_train, X_test, y_test):
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, pred_train),
        'balanced_accuracy': balanced_accuracy_score(y_train, pred_train),
        'recall': recall_score(y_train, pred_train),
        'f1_score': f1_score(y_train, pred_train)
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, pred_test),
        'recall': recall_score(y_test, pred_test),
        'f1_score': f1_score(y_test, pred_test)
    }

    return metrics_train, metrics_test


def generar_confusiones(modelo, X_train, y_train, X_test, y_test):
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    cm_train = confusion_matrix(y_train, pred_train)
    cm_test = confusion_matrix(y_test, pred_test)

    tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()
    tn_te, fp_te, fn_te, tp_te = cm_test.ravel()

    cm_dict_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {'predicted_0': int(tn_tr), 'predicted_1': int(fp_tr)},
        'true_1': {'predicted_0': int(fn_tr), 'predicted_1': int(tp_tr)}
    }

    cm_dict_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {'predicted_0': int(tn_te), 'predicted_1': int(fp_te)},
        'true_1': {'predicted_0': int(fn_te), 'predicted_1': int(tp_te)}
    }

    return cm_dict_train, cm_dict_test


def guardar_modelo(modelo, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(modelo, f)


if __name__ == "__main__":
    # Lectura de datos
    df_train = pd.read_csv("files/input/train_data.csv.zip")
    df_test = pd.read_csv("files/input/test_data.csv.zip")

    # Preprocesamiento
    df_train = procesar_datos(df_train)
    df_test = procesar_datos(df_test)

    X_train, y_train = df_train.drop(columns=['default']), df_train['default']
    X_test, y_test = df_test.drop(columns=['default']), df_test['default']

    pipeline = construir_pipeline()
    modelo_ajustado = ajustar_modelo(pipeline, 10, X_train, y_train, 'balanced_accuracy')

    os.makedirs("files/models", exist_ok=True)
    guardar_modelo(modelo_ajustado, "files/models/model.pkl.gz")

    resultados = []

    m_train, m_test = obtener_metricas(modelo_ajustado, X_train, y_train, X_test, y_test)
    resultados.extend([m_train, m_test])

    cm_train, cm_test = generar_confusiones(modelo_ajustado, X_train, y_train, X_test, y_test)
    resultados.extend([cm_train, cm_test])

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", 'w') as f_out:
        for item in resultados:
            f_out.write(json.dumps(item) + '\n')
