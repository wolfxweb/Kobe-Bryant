import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'


df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

# st.write(df_prod)
# st.write(df_dev)
st.title("Painel de Controle")
st.subheader("Engenharia de Machine Learning")
st.text("Exprimento utilizando a base de dados  https://www.kaggle.com/c/kobe-bryant-shot-selection/data")
fignum = plt.figure(figsize=(6,4))
# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax = plt.gca())

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             label='Producao',
             ax = plt.gca())



# User wine

plt.title('Monitoramento Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)


try:
    st.subheader("Classsification report")
    target_column = 'shot_made_flag'
    if target_column in df_dev.columns:
        report = metrics.classification_report(df_dev.prediction_label, df_dev[target_column], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
    else:
        st.write(f"Coluna  '{target_column}' não encontrada")
except KeyError as e:
    st.write(f"Error: {e}")