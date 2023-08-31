import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Загрузка данных о продающихся объектах
# Load data about selling objects
sell_data = pd.read_csv('data/sell_test.csv')
# Удаление строк с пропущенными значениями координат
# Drop rows with missing values of coordinates
sell_data.dropna(subset=['latitude'])

# Загрузка данных о сдающихся объектах в таблицу nympy для работы с библиотекой scikit-learn
# Load data about renting objects to numpy table for working with scikit-learn library
X = sell_data[['square', 'rooms', 'latitude', 'longitude']].to_numpy()
Y = sell_data[['price']].to_numpy()#.transpose()

# Разбиение данных на обучающую и тестовую выборки
# Split data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Обучение модели линейной регрессии, случайного леса и экстра-деревьев 
# Training linear regression, random forest models and extra trees
model = LinearRegression().fit(X_train, y_train)
model2 = RandomForestRegressor().fit(X_train, y_train.ravel())
model3 = ExtraTreesRegressor().fit(X_train, y_train.ravel())

# Предсказание цен на тестовой выборке
# Predict prices on test sample
test_predictions_linear = model.predict(X_test)
test_predictions_random_forest = model2.predict(X_test)
test_predictions_extra_trees = model3.predict(X_test)


# Визуализация результатов предсказания цен на тестовой выборке 
# для модели линейной регрессии
# Visualize results of prediction on test sample for linear regression model
plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_predictions_linear) # рисуем точки, соответствущие парам настоящее значение - прогноз
plt.plot([0, 3 * 10**6], [0, 3 * 10**6]) # рисуем прямую, на которой предсказания и настоящие значения совпадают
plt.xlabel('Настоящая цена', fontsize=20)
plt.ylabel('Предсказанная цена (LinearRegression)', fontsize=20)
plt.show()

# Визуализация результатов предсказания цен на тестовой выборке
# для модели случайного леса
# Visualize results of prediction on test sample for random forest model
plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_predictions_random_forest)
plt.plot([0, 4 * 10**6], [0, 4 * 10**6])
plt.xlabel('Настоящая цена', fontsize=20)
plt.ylabel('Предсказанная цена (RandomForestRegressor)', fontsize=20)
plt.show()

# Визуализация результатов предсказания цен на тестовой выборке
# для модели экстра-деревьев
# Visualize results of prediction on test sample for extra trees model
plt.figure(figsize=(7, 7))
plt.scatter(y_test, test_predictions_extra_trees)
plt.plot([0, 4 * 10**6], [0, 4 * 10**6])
plt.xlabel('Настоящая цена', fontsize=20)
plt.ylabel('Предсказанная цена (ExtraTreesRegressor)', fontsize=20)
plt.show()

# Расчет метрик качества моделей
# "средняя абсолютная ошибка", "средняя квадратичная ошибка" и "средняя абсолютная процентная ошибка"
# Calculation of quality metrics for models
# "mean absolute error", "mean squared error" and "mean absolute percentage error"

# Линейная регрессия
# Linear regression
mean_absolute_error_linear_model = mean_absolute_error(y_test, test_predictions_linear) 
mean_squared_error_linear_model = mean_squared_error(y_test, test_predictions_linear)
mean_absolute_percentage_error_linear_model = mean_absolute_percentage_error(y_test, test_predictions_linear)

# Случайный лес
# Random forest
mean_absolute_error_random_forest_model = mean_absolute_error(y_test, test_predictions_random_forest)
mean_squared_error_random_forest_model = mean_squared_error(y_test, test_predictions_random_forest)
mean_absolute_percentage_error_forest_model = mean_absolute_percentage_error(y_test, test_predictions_random_forest)

# Экстра-деревья
# Extra trees
mean_absolute_error_extra_trees_model = mean_absolute_error(y_test, test_predictions_extra_trees)
mean_squared_error_extra_trees_model = mean_squared_error(y_test, test_predictions_extra_trees)
mean_absolute_percentage_error_extra_trees_model = mean_absolute_percentage_error(y_test, test_predictions_extra_trees)

# Вывод результатов
# Print results
print("MAE: {0:7.2f}, RMSE: {1:7.2f}, MAPE: {2:7.2f} для модели линейной регрессии".format(
        mean_absolute_error_linear_model, 
        np.sqrt(mean_squared_error_linear_model), 
        mean_absolute_percentage_error_linear_model))

print("MAE: {0:7.2f}, RMSE: {1:7.2f}, MAPE: {2:7.2f} для модели случайного леса".format(
       mean_absolute_error_random_forest_model, 
       np.sqrt(mean_squared_error_random_forest_model), 
       mean_absolute_percentage_error_forest_model))

print("MAE: {0:7.2f}, RMSE: {1:7.2f}, MAPE: {2:7.2f} для модели экстра-деревьев".format(
        mean_absolute_error_extra_trees_model,
        np.sqrt(mean_squared_error_extra_trees_model),
        mean_absolute_percentage_error_extra_trees_model))

# Добавим в таблицу данные о предсказанных ценах и разнице между предсказанными и реальными ценами
# в абсолютном значении для модели случайного леса и отсортируем, чтобы найти аномальные объекты
# Add to table data about predicted prices and difference between predicted and real prices
# in absolute value for random forest model and sort to find anomalous objects
sell_data['predicted_price'] = model2.predict(X)
sell_data['difference'] = abs(sell_data['price'] - sell_data['predicted_price'])
sell_data.sort_values(by=['difference'], ascending=False, inplace=True)

# Выводим 20 объектов с наибольшей разницей между предсказанными и реальными ценами
# с идентификаторами объявлений, чтобы посмотреть, что с ними не так
# Print 20 objects with the largest difference between predicted and real prices
# with links to ads to see what's wrong with them
print(sell_data[['field_1', 'difference']].head(20))

# Удаляем из таблицы 20 аномальных объектов и заново обучаем модель случайного леса
# Drop 20 anomalous objects from table and train random forest model again
sell_data.drop(sell_data[['field_1', 'difference']].head(20).index, inplace=True)
# Печатаем размер таблицы после удаления аномальных объектов
# Print size of table after drop anomalous objects
print(sell_data.shape)
X = sell_data[['square', 'rooms', 'latitude', 'longitude']].to_numpy()
Y = sell_data[['price']].to_numpy()#.transpose()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
model2 = RandomForestRegressor().fit(X_train, y_train.ravel())
test_predictions_random_forest = model2.predict(X_test)
mean_absolute_error_random_forest_model = mean_absolute_error(y_test, test_predictions_random_forest)
mean_squared_error_random_forest_model = mean_squared_error(y_test, test_predictions_random_forest)
mean_absolute_percentage_error_forest_model = mean_absolute_percentage_error(y_test, test_predictions_random_forest)
# Вывод результатов
print("MAE: {0:7.2f}, RMSE: {1:7.2f}, MAPE: {2:7.2f} для модели случайного леса после удаления аномальных объектов".format(
        mean_absolute_error_random_forest_model,
        np.sqrt(mean_squared_error_random_forest_model),
        mean_absolute_percentage_error_forest_model))



