import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

st.title("Painel de Controle")
st.subheader("Engenharia de Machine Learning")
st.markdown("""
        Em homenagem ao jogador da NBA Kobe Bryant (falecido em 2020), foram disponibilizados os dados de 20 anos de arremessos, bem sucedidos ou não, e informações correlacionadas.
O objetivo desse estudo é aplicar técnicas de inteligência artificial para prever se um arremesso será convertido em pontos ou não. 
ata""")
fignum = plt.figure(figsize=(6,4))
# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax=plt.gca())

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             label='Producao',
             ax=plt.gca())

plt.title('Monitoramento Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)

try:
    st.subheader("Classification report base desenvolvimento")
    target_column = 'shot_made_flag'
    if target_column in df_dev.columns:
        report = metrics.classification_report(df_dev.prediction_label, df_dev[target_column], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
    else:
        st.write(f"Coluna  '{target_column}' não encontrada")

    st.subheader("Matriz de Confusão")
    cm = metrics.confusion_matrix(df_dev.prediction_label, df_dev[target_column])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Rótulos Previstos')
    ax.set_ylabel('Rótulos Verdadeiros')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

except KeyError as e:
    st.write(f"Error: {e}")
