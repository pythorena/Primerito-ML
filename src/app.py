#from utils import db_connect
#engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Problema -> algoritmo de clasificación que ayude a predecir si un cliente contratará o no un depósito a largo plazo.

#EDA
# Cargar conjunto de datos
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep = ";")
total_data.head()

total_data.shape

total_data.info()

# Eliminación de duplicados
if total_data.duplicated().sum():
    total_data = total_data.drop_duplicates()
print(total_data.shape)
total_data.head()

# Eliminación de información irrelevante 
total_data.drop(["nr.employed"], axis = 1, inplace = True)
total_data.head()

# Factorización de variables numéricas
total_data["y_n"] = pd.factorize(total_data["y"])[0]
total_data["poutcome_n"] = pd.factorize(total_data["poutcome"])[0]

'''
Tras los análisis univariante y multivariante, además de las matrices de correlación 
llego a la conclusión de elegir las siguientes variables como predictoras:
"duration", "pdays","euribor3m","emp.var.rate","poutcome_n"
'''

total_data.describe()

# Eliminación de nulos
total_data.isnull().sum().sort_values(ascending=False)

# División del conjunto en train y test
from sklearn.model_selection import train_test_split

num_variables = ["duration", "pdays","euribor3m","emp.var.rate","poutcome_n"]

X = total_data.drop("y_n", axis = 1)[num_variables]
y = total_data["y_n"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Escalonado MinMax (he probado a normalizar y el resultado es ligeramente mejor con este escalonado)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = num_variables)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = num_variables)

X_train_scal.head()

# Entrenamiento del modelo
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scal, y_train)

y_pred = model.predict(X_test_scal)

# Búsqueda por malla (he probado la aleatoria y sale ligeramente mejor por malla)

from sklearn.model_selection import GridSearchCV

# Definimos los parámetros a mano que queremos ajustar
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

# Inicializamos la grid
grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 5)
grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train_scal, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")

model_grid = LogisticRegression(penalty = "l2", C = 100, solver = "newton-cg")
model_grid.fit(X_train_scal, y_train)
y_pred = model_grid.predict(X_test_scal)

grid_accuracy = accuracy_score(y_test, y_pred)
grid_accuracy

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

f1_score(y_test, y_pred)

banco_cm = confusion_matrix(y_test, y_pred)

# Dibujaremos esta matriz para hacerla más visual
cm_df = pd.DataFrame(banco_cm)

plt.figure(figsize = (3, 3))
sns.heatmap(cm_df, annot=True, fmt="d", cbar=False)

plt.tight_layout()

plt.show()