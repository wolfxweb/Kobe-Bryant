import pandas as pd
import numpy as np
import pycaret.classification as pc
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Para usar o sqlite como repositorio
mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = 'ProjetoKobe '
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

# Carregar o modelo
model_uri = f"models:/kobe@staging"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Carregar os dados de produção
data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')

# Remover a coluna 'shot_made_flag' dos dados de produção
colunas_selecionadas = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
X_prod = data_prod[colunas_selecionadas]

with mlflow.start_run(experiment_id=experiment_id, run_name='PipelineAplicacao'):
    # Fazer previsões
    Y_pred_proba = loaded_model.predict_proba(X_prod)[:, 1]
    
    # Adicionar as previsões aos dados de produção
    data_prod['predict_score'] = Y_pred_proba

    # Salvar os dados de produção com as previsões
    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')

    print(data_prod)
