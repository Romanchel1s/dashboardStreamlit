import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/output_card_trans.csv')

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
