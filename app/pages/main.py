import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

uploaded_file = st.file_uploader("Выберите файл датасета")

if uploaded_file is not None:
    X_test = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop(['Unnamed: 0'], axis=1)
        X_test = X_test.drop(['fraud'], axis=1 )
    st.write("Загруженный датасет:", X_test)

else:
    st.title("Получить предсказания мошеннической транзакции.")


    st.header("Distance from home")
    distance_from_home = st.number_input("Число:", value = 10.8299)

    st.header("Distance from last transaction")
    distance_from_last_transaction = st.number_input("Число:", value = 0.1756)

    st.header("Ratio to median purchase price")
    ratio_to_median_purchase = st.number_input("Число:", value = 1.1756)

    st.header("Repeat retailer")
    repeat_relailer = st.number_input("Число 0 или 1:", value = 1)

    st.header("Used chip")
    used_chip = st.number_input("Число 0 или 1:", value = 0)

    st.header("Used pin number")
    used_pin_number = st.number_input("Число 1 или 0:", value = 1)

    st.header("Online order")
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
    with open('../models/KNN_model.pkl', 'rb') as file:
        knn_model = joblib.load(file)

    with open('../models/Kmeans_model.pkl', 'rb') as file:
         kmeans_model= joblib.load(file)
    
    with open('../models/Stacking_model.pkl', 'rb') as file:
        stacking_model = joblib.load(file)
    
    with open('../models/Bagging_model.pkl', 'rb') as file:
        bagging_model = joblib.load(file)
    
    with open('../models/GradientBoostingClassifier_model.pkl', 'rb') as file:
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


    model_class = tf.keras.models.load_model('../models/ClassificationModel')
    st.header("Neural network:")
    nn_pred = model_class.predict(X_test)[0]
    y_pred = tf.argmax(nn_pred, axis=0)
    st.write(f"{y_pred}")