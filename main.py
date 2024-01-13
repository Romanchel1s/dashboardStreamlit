import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf


# Загрузка датасета
data = pd.read_csv('data/output_card_trans.csv')
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)
df = data
X = data.drop(['fraud'], axis=1)
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



st.title('Расчётно графичесикая работа ML')
# Навигация
st.sidebar.title('Навигация:')
page = st.sidebar.radio(
    "Выберите страницу",
    ("Разработчик", "Датасет", "Визуализация", "Инференс модели")
)
# Информация о разработчике
def page_developer():
    st.title("Информация о разработчике")
    st.header("Тема РГР:")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Фотография")
        st.image("data/photo.jpg", width=150)  # Укажите путь к вашей фотографии
    
    with col2:
        st.header("Контактная информация")
        st.write("ФИО: Пуртов Роман Сергеевич")
        st.write("Номер учебной группы: ФИТ-221")
    
    

# Информаиця о нашем датасете
def page_dataset():
    st.title("Информация о наборе данных")

    st.header("Тематика датасета: Операции банковской карты")
    st.header("Описание признаков:")
    st.write("- distance_from_home: расстояние от дома владельца карты")
    st.write("- distance_from_last_transaction: расстояние от последней транзакции с карты")
    st.write("- ratio_to_median_purchase_price: различие от средней цены покупки")
    st.write("- repeat_retailer: совершались ли транзакции у этого ритейлера раньше")
    st.write("- used_chip: использовался ли чип")
    st.write("- used_pin_number: использовался ли пинкод")
    st.write("- online_order: была ли покупка онлайн")
    st.header(" Предобработка данных:")
    st.write("В датасете были пропущенные значения. Они были удалены")
    st.write("Были удалены дубликаты")
    st.write("Устранён дисбаланс классов")
    
    
# Страница с визуализацией
def page_data_visualization():
    st.title("Визуализации данных")

    st.write(df)

    st.title("Датасет банковские операции")

    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(12, 8))
    selected_cols = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
    selected_df = df[selected_cols]
    sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)

    st.header("Гистограммы")

    columns = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df.sample(5000)[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)

    st.header("Ящики с усами ")
    outlier = df[columns]
    Q1 = outlier.quantile(0.25)
    Q3 = outlier.quantile(0.75)
    IQR = Q3-Q1
    data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]


    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data_filtered[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)

    st.header("Круговая диаграмма целевого признака")
    plt.figure(figsize=(8, 8))
    df['fraud'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('fraud')
    plt.ylabel('')
    st.pyplot(plt)


# Страница с инференсом моделей
def page_predictions():
    st.title("Предсказания моделей машинного обучения")

    uploaded_file = st.file_uploader("Выберите файл датасета")

    if uploaded_file is not None:
        X_test = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in X_test.columns:
            X_test = X_test.drop(['Unnamed: 0'], axis=1)
            X_test = X_test.drop(['fraud'], axis=1 )
        st.write("Загруженный датасет:", X_test)

    else:
        st.title("Получить предсказание мошеннической транзакции.")


        st.header("Расстояние от дома (км)")
        distance_from_home = st.number_input("Число:", value = 10.8299)

        st.header("Расстояние от последней транзакции (км)")
        distance_from_last_transaction = st.number_input("Число:", value = 0.1756)

        st.header("Отличие покупки от медианной")
        ratio_to_median_purchase = st.number_input("Число:", value = 1.1756)

        st.header("Повторный продавец")
        repeat_relailer = st.number_input("Число 0 или 1:", value = 1)

        st.header("С использованием чипа")
        used_chip = st.number_input("Число 0 или 1:", value = 0)

        st.header("С импользованием пин-кода")
        used_pin_number = st.number_input("Число 1 или 0:", value = 1)

        st.header("Онлайн заказ")
        online_order = st.number_input("Число 1 или 0:", value = 0)


        X_test = pd.DataFrame({'distance_from_home': [distance_from_home],
                            'distance_from_last_transaction': [distance_from_last_transaction],
                            'ratio_to_median_purchase_price': [ratio_to_median_purchase],
                            'repeat_retailer': [repeat_relailer],
                            'used_chip': [used_chip],
                            'used_pin_number': [used_pin_number],
                            'online_order': [online_order],         
                            })


    button_clicked = st.button("Предсказать")

    if button_clicked:
        with open('models/KNN_model.pkl', 'rb') as file:
            knn_model = joblib.load(file)

        with open('models/KMeans_model.pkl', 'rb') as file:
            kmeans_model= joblib.load(file)
        
        with open('models/Stacking_model.pkl', 'rb') as file:
            stacking_model = joblib.load(file)
        
        with open('models/Bagging_model.pkl', 'rb') as file:
            bagging_model = joblib.load(file)
        
        with open('models/GradientBoostingClassifier_model.pkl', 'rb') as file:
            gradboost_model = joblib.load(file)

                

        st.header("KNN:")
        pred =[]
        knn_pred = knn_model.predict(X_test)[0]
        pred.append(int(knn_pred))
        st.write(f"{knn_pred}")

        st.header("KMeans:")
        pred =[]
        knn_pred = kmeans_model.predict(X_test)[0]
        pred.append(int(knn_pred))
        st.write(f"{knn_pred}")

        st.header("Stacking (LinearRegressiong, DecisionTreeRegressor):")
        pred =[]
        knn_pred = stacking_model.predict(X_test)[0]
        pred.append(int(knn_pred))
        st.write(f"{knn_pred}")

        st.header("Bagging (DecisionTreeClassifier):")
        pred =[]
        knn_pred = bagging_model.predict(X_test)[0]
        pred.append(int(knn_pred))
        st.write(f"{knn_pred}")

        st.header("Gradient Boosting Classifier:")
        pred =[]
        knn_pred = gradboost_model.predict(X_test)[0]
        pred.append(int(knn_pred))
        st.write(f"{knn_pred}")


        model_class = tf.keras.models.load_model('models/ClassificationModel')
        st.header("Neural network:")
        nn_pred = model_class.predict(X_test)[0]
        y_pred = tf.argmax(nn_pred, axis=0)
        st.write(f"{y_pred}")


if page == "Разработчик":
    page_developer()
elif page == "Датасет":
    page_dataset()
elif page == "Инференс модели":
    page_predictions()
elif page == "Визуализация":
    page_data_visualization()
