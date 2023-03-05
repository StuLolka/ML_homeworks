import pandas as pd # Для работы с данными
from sklearn.preprocessing import LabelEncoder
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt  # Библиотека для визуализации результатов

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('adult.csv', delimiter = ',')
print(f"Sum data null:\n{data.isnull()}")
print(f"Sum data null:\n{data.isnull().sum()}\n\n")
print(f"\nData:\n{data.head(10)}\n\n")
print("\nData info:\n")
data.info()

print(f"\nMean by income:\n{data.groupby('income').mean()}\n\n")
# Можно заметить, что сильно отличаются capital-gain и capital-loss

# Сделаем так, чтобы 'income' принимал одну из двух значений: 0 или 1
le = LabelEncoder()
le.fit(data['income'])
data["income"] = le.transform(data['income'])

# Визуализируем несколько предполагаемых зависимостей
for i in ['age', 'hours-per-week', 'capital-loss', 'capital-gain']:
    less = list(data[data["income"] == 0][i])
    more = list(data[data["income"] == 1][i])
    xmin = min(min(less), min(more))
    xmax = max(max(less), max(more))
    width = (xmax - xmin) / 40
    sns.distplot(less, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(more, color='b', kde=False, bins=np.arange(xmin, xmax, width))
    plt.title(f'Histogram for {i}')
    plt.legend(["<=50K", ">50K"])
    plt.show()


# Выбираем "нужную" информацию
selected_data = data[['age', 'hours-per-week', 'marital-status', 'native-country', 'education', 'capital-loss', 'capital-gain', 'workclass']]


x = pd.get_dummies(selected_data, columns = ['marital-status', 'native-country', 'education', 'workclass'])
y = data["income"]


# Создаем модели
# Если не масштабировать данные (StandardScaler()) - результаты будут хуже на обеих моделях
lrModel = make_pipeline(StandardScaler(), LogisticRegression())
svcModel = make_pipeline(StandardScaler(), SVC())

# Разделяем данные на train и test, на тест отдаем 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

# Обучаем модель
lrModel.fit(x_train, y_train)
svcModel.fit(x_train, y_train)

# Смотрим точность
print(f"\n\nLogisticRegression train score = {lrModel.score(x_train, y_train)}") # 0.8452895861592404
print(f"\nLogisticRegression test score = {lrModel.score(x_test, y_test)}") # 0.8483979936533934

print(f"\nSVC train score = {svcModel.score(x_train, y_train)}") # 0.8489238092800655
print(f"\nSVC test score = {svcModel.score(x_test, y_test)}\n\n") # 0.8474767120483161