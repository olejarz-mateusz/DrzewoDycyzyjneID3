import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Otworzenie pliku z danymi
colNames=['target','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population','habitat']
mushroom_data = pd.read_csv('agaricus-lepiota.data', sep=',', names=colNames)

# Wyświetlenie informacji o zbiorze
print("Training dataset:")
print("patients_train_data:", mushroom_data.shape)

# Tworzymy z niego dwa kolejne zestawy w celu modyfikacji kolumny z brakujacymi wartościami
mushroom_get_column_mode = mushroom_data.copy()
mushroom_delete_column = mushroom_data.copy()

# Brakujące wartości 11 kolumny zastępujemy poprzez usunięcie wierszy, w których występuje znak zapytania lub zastąpienie ich najczęściej występującą wartością. Niestety nie udało się wykonać uzupełnienia innymi funkcjami statystycznymi, jak mediana lub średnia. Sprawdzimy też co stanie się po zostawieniu znaków zapytania.
mode_value = mushroom_data.mode().iloc[:,11]

mushroom_get_column_mode.replace("?", np.nan, inplace=True)
mushroom_get_column_mode.replace(np.nan, mode_value[0], inplace=True)
mushroom_delete_column.replace("?", np.nan, inplace=True)
mushroom_delete_column.dropna(inplace=True)

# Zamieniamy dane z tekstu na liczby, w przeciwnym wypadku nie zadziała DecisionTreeClassifier
le = preprocessing.LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = le.fit_transform(mushroom_data[column])
for column in mushroom_delete_column.columns:
    mushroom_delete_column[column] = le.fit_transform(mushroom_delete_column[column])
for column in mushroom_get_column_mode.columns:
    mushroom_get_column_mode[column] = le.fit_transform(mushroom_get_column_mode[column])

mushroom_missing_left_x = mushroom_data.iloc[:, 1:23]
mushroom_column_mode_x = mushroom_get_column_mode.iloc[:, 1:23]
mushroom_delete_column_x = mushroom_delete_column.iloc[:, 1:23]
mushroom_missing_left_y = mushroom_data.iloc[:, 0]
mushroom_column_mode_y = mushroom_get_column_mode.iloc[:, 0]
mushroom_delete_column_y = mushroom_delete_column.iloc[:, 0]

missing_left_train_x, missing_left_test_x, missing_left_train_y, missing_left_test_y = \
    train_test_split(mushroom_missing_left_x, mushroom_missing_left_y, train_size=0.8, random_state=1)
mode_train_x, mode_test_x, mode_train_y, mode_test_y = \
    train_test_split(mushroom_column_mode_x, mushroom_column_mode_y, train_size=0.8, random_state=1)
deleted_train_x, deleted_test_x, deleted_train_y, deleted_test_y = \
    train_test_split(mushroom_delete_column_x, mushroom_delete_column_y, train_size=0.8, random_state=1)

left_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, class_weight='balanced', random_state=1)
mode_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, class_weight='balanced', random_state=1)
deleted_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, class_weight='balanced', random_state=1)

left_model.fit(missing_left_train_x, missing_left_train_y)
mode_model.fit(mode_train_x, mode_train_y)
deleted_model.fit(deleted_train_x, deleted_train_y)

# Wyniki poszczególnych modeli na różnych danych testowych
print(left_model.score(missing_left_test_x, missing_left_test_y))
print(left_model.score(mode_test_x, mode_test_y))
print(left_model.score(deleted_test_x, deleted_test_y))
print(mode_model.score(missing_left_test_x, missing_left_test_y))
print(mode_model.score(mode_test_x, mode_test_y))
print(mode_model.score(deleted_test_x, deleted_test_y))
print(deleted_model.score(missing_left_test_x, missing_left_test_y))
print(deleted_model.score(mode_test_x, mode_test_y))
print(deleted_model.score(deleted_test_x, deleted_test_y))

# Wykres wyników poszczególnych modeli
test_scores_left = []
train_scores_left = []
for i in range(1, 10):
    graph1 = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=i)
    graph1.fit(missing_left_train_x, missing_left_train_y)
    test_scores_left.append(graph1.score(missing_left_test_x, missing_left_test_y))
    train_scores_left.append(graph1.score(missing_left_train_x, missing_left_train_y))
plt.plot(test_scores_left, color='red')
plt.plot(train_scores_left)
plt.show()

test_scores_mode = []
train_scores_mode = []
for i in range(1, 10):
    graph2 = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=i)
    graph2.fit(mode_train_x, mode_train_y)
    test_scores_mode.append(graph2.score(mode_test_x, mode_test_y))
    train_scores_mode.append(graph2.score(mode_train_x, mode_train_y))
plt.plot(test_scores_mode, color='red')
plt.plot(train_scores_mode)
plt.show()

test_scores_deleted = []
train_scores_deleted = []
for i in range(1, 10):
    graph2 = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=i)
    graph2.fit(deleted_train_x, deleted_train_y)
    test_scores_deleted.append(graph2.score(deleted_test_x, deleted_test_y))
    train_scores_deleted.append(graph2.score(deleted_train_x, deleted_train_y))
plt.plot(test_scores_deleted, color='red')
plt.plot(train_scores_deleted)
plt.show()

# Macierze konfuzji poszególnych modeli
left_model_predictions = left_model.predict(missing_left_test_x)
left_model_cnf_matrix = confusion_matrix(missing_left_test_y, left_model_predictions)
sn.heatmap(left_model_cnf_matrix)

mode_model_predictions = mode_model.predict(mode_test_x)
mode_model_cnf_matrix = confusion_matrix(mode_test_y, mode_model_predictions)
sn.heatmap(mode_model_cnf_matrix)

deleted_model_predictions = deleted_model.predict(deleted_test_x)
deleted_model_cnf_matrix = confusion_matrix(deleted_test_y, deleted_model_predictions)
sn.heatmap(deleted_model_cnf_matrix)