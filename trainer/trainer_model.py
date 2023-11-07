# Databricks notebook source
# MAGIC %md
# MAGIC ##### Notebook para treinar o modelo
# MAGIC

# COMMAND ----------

from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, date_format, min, udf, concat, unix_timestamp, format_string, map_values, substring_index
from pyspark.sql.types import IntegerType, DecimalType, DoubleType, StringType
from pyspark.sql.session import SparkSession

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import mlflow.xgboost
mlflow.xgboost.autolog()

spark = SparkSession.builder.getOrCreate()

import mlflow  # type: ignore
from sklearn.pipeline import Pipeline
#from pyspark.ml import Pipeline
#from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame

# COMMAND ----------

# # ---------------------------------------------
# # Base para modelagem
# database = spark.table(DATA_PREPARATION_TABLE_NAME).toPandas()
# # ---------------------------------------------

# COMMAND ----------

def _load_train_data() -> DataFrame:
#def _load_train_data(dt_inicio: str, dt_fim: str) -> DataFrame:
    """
    Esta funcao:
        - Realiza o carregamento das features que serao utilizadas no modelo.

    Importane:
        - Os dados vem prontos do arquivo da funcao '_data_transformation()'
    do arquivo 'data_preparation/data_preparation, quando retornados, são salvos
    para o caso da 'persistência' e segue o código.

    Dados de Entrada:
                    - dt_inicio: String que representa a data inicial para retorno da base
                    - dt_fim: String que representa a daa final para retorno da base
    Dados de saida:
                    - features: DataFrame de features do modelo
    """

    features = database  # verificacao_existencia_tabela(dt_inicio, dt_fim);
    
    return features


# COMMAND ----------

def _remove_high_correlation(dataframe: DataFrame) -> DataFrame:  # pylint: disable=W0613
    """
    Esta funcao:
        - Remove colunas de features com alta correlação

    Importante:
        - Valor de alta correlação definida foi acima de: 0.7

    Dados de Entrada:
                    - dataframe: DataFrame que possui as variaveis
    Dados de saida:
                    - dataframe: DataFrame com as colunas que possuem alta correlação removidas
    """
    
    # ------
    base_xgb = database.copy()
    # ------
    colunas_drop = [coluna for coluna in base_xgb.columns if "install_channel" in coluna]
    # ------
    xgb_v1 = base_xgb.drop(colunas_drop, axis=1).copy()
    # ------
    correlation_matrix = xgb_v1.corr()
    # ------
    highly_correlated = (correlation_matrix.abs() >= 0.7) & (correlation_matrix.abs() < 1.0)

    features_to_remove = set()
    for i in range(len(highly_correlated.columns)):
        for j in range(i):
            if highly_correlated.iloc[i, j]:
                colname = highly_correlated.columns[i]
                features_to_remove.add(colname)

    base_xgb_teste = xgb_v1.copy().drop(features_to_remove, axis=1)

    base_xgb_teste['uninstall'] = base_xgb_teste['uninstall'].astype(int)

    return base_xgb_teste



# COMMAND ----------

def _split_train_data(dataframe: DataFrame, random_state: int = 42):  # pylint: disable=W0613
    """
    Esta funcao:
        - Recebe um DataFrame e divide os dados em
        conjuntos treino e teste, com tamanhos proporcionais
        definido pelo usuario

    Importante:
        - A principio estamos dividindo 0.7 e 0.3, para treino e teste

    Dados de Entrada:
                    - dataframe: DataFrame que possui as variaveis de clientes, mediana de horario e desvio padrao das suas compras
    Dados de saida:
                    - df_treino: DataFrame que sera utilizado para o treinamento do modelo
                    - df_teste: DataFrame que sera utilizado para o teste do modelo
    Variaveis:
                    - [0.7, 0.3]: que definem em proporcao o tamanho da amostra de treinamento e teste  
    """

    X = dataframe.drop(['uninstall', 'id_ga'], axis=1)
    y = dataframe['uninstall']

    #df_treino, df_teste = dataframe.randomSplit([0.7, 0.3], seed = random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test #df_treino, df_teste

# COMMAND ----------

def _create_model_pipeline(y_train: DataFrame) -> Pipeline:
    """
    Esta funcao:
        - Realiza a criacao da pipeline do modelo.

    Importante:
        - Foi adicionado o modelo a ser utilizado, assim como o
        vectorAssembler e a criacao do pipeline

    Dados de Entrada:
                    - NULL
    Dados de saida:
                    - pipeline: Pipeline gerado de acordo com o que se é necessário para o modelo
    Variaveis:
                    - k: na funcao KMeans, esse valor é referente a quantidade de grupos que se deseja gerar com o algoritmo
    """

    proporcao_classe_positiva = np.sum(y_train == 0) / np.sum(y_train == 1)

    best_xgb_classifier = xgb.XGBClassifier(random_state=42, scale_pos_weight=proporcao_classe_positiva,
                                            learning_rate =  0.2, max_depth = 6, n_estimators = 300, subsample = 0.8)

    #pipeline = Pipeline(stages = [best_xgb_classifier])  # type: ignore
    #pipeline = Pipeline([best_xgb_classifier])
    pipeline = Pipeline([('xgboost', best_xgb_classifier)])

    return pipeline

# COMMAND ----------

def _train_model(pipeline: Pipeline, X_train, y_train):
    """
    Esta funcao:
        - Treina a pipeline.

    Dados de Entrada:
                    - pipeline: Pipeline gerado no "_create_model_pipeline"
                    - data: DataFrame com os dados de treinamento
    Dados de saida:
                    - PipelineModel: Modelo treinado via Pipeline
    """

    return pipeline.fit(X_train, y_train)

# COMMAND ----------

def _evaluate_pipeline(dataframe: DataFrame, y_test, y_pred) -> Dict[str, float]:
    """
    Esta funcao:
        - Calcula a metrica de avaliacao do modelo.

    Dados de Entrada:
                    - dataframe: DataFrame que sera utilizado para calcular a metrica de avaliação.
                        Para que a metrica seja calculada com sucesso, o DataFrame precisa ter as colunas obrigatorias labels e prediction.
    Dados de saida:
                    - metrics_result: Metricas de avaliacao caculadas.
    """

    report = classification_report(y_test, y_pred)
    report_lines = report.split('\n')
    
    # Obter os valores de precisão e acurácia da primeira classe
    precision_first_class = float(report_lines[2].split()[1])
    precision_second_class = float(report_lines[3].split()[1])
    accuracy = float(report_lines[-2].split()[2])

    metrics_result = {"Acurácia": accuracy, "Precisao - Classe 0": precision_first_class, "Precisao - Classe 1": precision_second_class}

    return metrics_result

# COMMAND ----------

# def _extract_spark_model_params(model) -> dict:
#     """
#     Esta funcao:
#         - Extrai o parametros de um modelo spark treinado.

#     Dados de Entrada:
#                     - model: Modelo treinado que sera utilizado para extrair os parametros
#     Dados de saida:
#                     - metrics_result: Dicionario que contem os parametros do modelo treinado.
#     """

#     return {key.name: value for key, value in model.stages[-1].extractParamMap().items()}

# COMMAND ----------

def _save_pipeline(pipeline) -> None:
    """
    Esta funcao:
        -  Realiza o versionamento do modelo.

    Dados de Entrada:
                    - pipeline: Pipeline do modelo treinado para versionamento
    Dados de saida:
                    - NULL
    """

    #mlflow.spark.log_model(pipeline, MODEL_NAME, registered_model_name = MODEL_NAME)
    mlflow.sklearn.log_model(pipeline.named_steps['xgboost'], MODEL_NAME, registered_model_name = MODEL_NAME)
    

# COMMAND ----------

def train(dataInicio: str, dataFinal: str, experiment_id: int = EXPERIMENT_ID) -> None:
    """
    Esta funcao:
        - Orquestra o treinamento do modelo.
    
    Importante:
        - A data final da base de dados é um dia anterior ao definido, exemplo: se definido dataFinal = '2022-02-07' a base retornará daddos até '2022-02-06'

    Dados de Entrada:
                    - São repassadas como variáveis de argumento utilizando o argparse
                    - experiment_id: ID do experimento no MlFlow

    Dados de saida:
                    - NULL
    """

    #with mlflow.start_run(experiment_id=experiment_id):
    # Registre as métricas com o MLflow
    with mlflow.start_run():
        
        logger.info("Carregando base de treino")

        dataframe = _load_train_data()
        #dataframe = _load_train_data(dt_inicio = dataInicio, dt_fim = dataFinal)

        dataframe = _remove_high_correlation(dataframe)

        #train_data, test_data = _split_train_data(dataframe)
        X_train, X_test, y_train, y_test = _split_train_data(dataframe)

        logger.info("Criando pipeline e treinando o modelo")
        
        pipeline = _create_model_pipeline(y_train)
        
        model_fitted = _train_model(pipeline, X_train, y_train)

        logger.info("Fazendo predição da base de teste")
        
        y_pred = model_fitted.predict(X_test)
        metrics_result = _evaluate_pipeline(X_test, y_test, y_pred)

        logger.info("Salvando pipeline, metricas e parâmetros")
        
        ## Salve o modelo treinado
        _save_pipeline(model_fitted)
        
        #mlflow.log_param('learning_rate', model_fitted.named_steps['xgboost'].get_params()['learning_rate'])
        mlflow.log_metrics(metrics_result)      
        
    logger.info("Processo finalizado")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### Funções Úteis - logger

# COMMAND ----------

import logging

# DBTITLE 1,Logging
def structured_logging_info(suppress_spark: bool = True):
    """ Define a clear structure to the logging, and suppress the spark py4j if needed
    Args:
        suppress_spark: Boolean value informing if the logging of py4j will be in the warning level
    Returns:
        logger: The logger for the module
    """
    # Supress the INFO logging of spark python for java
    loggerSpark = logging.getLogger('py4j')
    loggerSpark.setLevel('WARNING')

    logger = logging.getLogger()

    # Formate the logger
    formatter = logging.Formatter('Line %(lineno)d : [ %(asctime)s ] %(filename)s/%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Configure stream handler for the cells
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(chandler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = structured_logging_info()