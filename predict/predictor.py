# Databricks notebook source
# MAGIC %md
# MAGIC ##### Notebook para fazer predições com o modelo
# MAGIC

# COMMAND ----------

from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, date_format, min, udf, concat, unix_timestamp, format_string, map_values, substring_index
from pyspark.sql.types import IntegerType, DecimalType, DoubleType, StringType
from pyspark.sql.session import SparkSession

import mlflow  # type: ignore
from sklearn.pipeline import Pipeline
#from pyspark.ml import Pipeline
#from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace

import mlflow.pyfunc
import mlflow.sklearn
import xgboost as xgb
import pandas as pd

# COMMAND ----------

def load_pipeline(model_uri: str) -> Pipeline:
    """
    Esta funcao:
        - Realiza o carregamento do modelo de machine learning.

    Dados de Entrada:
                    - model_uri: URI em que o modelo esta versionado no MLFlow.
    Dados de saida:
                    - pipeline: Modelo versionado carregado.
    """

    return mlflow.sklearn.load_model(model_uri)


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

def predict(data: DataFrame, pipeline):
    """
    Esta funcao:
        -  Realiza o carregamento do modelo de machine learning.

    Dados de Entrada:
                    - data: DataFrame Spark que sera utilizado para a modelagem e predicao
                    - pipeline: pipeline com o modelo que sera utilizado neste caso

    Dados de saida:
                    - prediction: DataFrame com a predicao do modelo
    """
    
    y_proba = pipeline.predict_proba(data)[:, 1]
    return y_proba


# COMMAND ----------

def adicao_da_coluna(data: DataFrame, Result) -> DataFrame:
    """
    Esta funcao:
        -  Adiciona a coluna de prob_uninstall no DataFrame.

    Dados de Entrada:
                    - data: DataFrame que sera utilizado para a modelagem e predicao
                    - Resul: vetor de probabilidades para uninstall

    Dados de saida:
                    - data: DataFrame com a adição da coluna de melhor horário
    """

    data.insert(1, 'prob_uninstall', Result)
    data = data[["id_ga", "prob_uninstall"]]
    data_spark = spark.createDataFrame(data)

    return data_spark


# COMMAND ----------

def _add_DtRef_TsAtualizacao_dataframe(df: DataFrame, dt_fim: str) ->  DataFrame:

    """
    Esta funcao:
        - Adiciona colunas referências de Ano, Mês e Dia e uma com 'datetime' atual.

    Importante:
        - Esta considerando o horario como UTC-3
        - Apenas como organização, está sendo utilizado um select para organização das colunas 
        
    Dados de Entrada:
                    - df: DataFrame após o ETL para inclusão das colunas
                    - dt_fim: String que representa a data final para retorno da base que será utilizado como referência

    Dados de saida:
                    - df: DataFrame com adição das colunas de Ano, Mês e Dia e 'datetime' atual

    """

    ano, mes, dia = dt_fim.split("-")
    df = df.withColumn("DtRef", lit(dt_fim))
    df = df.withColumn("AaParticao", lit(ano))
    df = df.withColumn("MmParticao", lit(mes))
    df = df.withColumn("DdParticao", lit(dia))

    df = df.withColumn("TsAtualizacao", lit(datetime.today()))

    return df

# COMMAND ----------

def save_predict_data(data: DataFrame, dt_fim: str) -> None:
    """
    Esta funcao:
        -  Realiza o salvamento do DataFrame com os valores preditos

    Dados de Entrada:
                    - data: DataFrame Spark com os resultados obtidos
                    - dt_fim: String que representa a data final para retorno da base
                    
    Dados de saida:
                    - NULL
    """
    
    flg_tb_existe = spark._jsparkSession.catalog().tableExists(PREDICTION_RESULT_TABLE_NAME.split(".")[0], PREDICTION_RESULT_TABLE_NAME.split(".")[1])
    if flg_tb_existe:

        spark.sql(f"DELETE FROM {PREDICTION_RESULT_TABLE_NAME} WHERE AaParticao = year('{dt_fim}') AND MmParticao = month('{dt_fim}') AND DdParticao = dayofmonth('{dt_fim}')")
        
    
    
    data.write.format("delta").partitionBy('AaParticao', 'MmParticao', 'DdParticao').option("mergeSchema", "true").mode("append").saveAsTable(PREDICTION_RESULT_TABLE_NAME)#, path = PREDICTION_RESULT_PATH)


# COMMAND ----------

def run(dataInicio: str, dataFinal: str, MODEL_PATH) -> None:
    """
    Esta funcao:
        -  Realiza a execucao da scoragem do modelo.
    
    Importante:
        - A data final da base de dados é um dia anterior ao definido, exemplo: se definido dataFinal = '2022-02-07' a base retornará daddos até '2022-02-06'
        
    Dados de Entrada:
                    - São repassadas como variáveis de argumento utilizando o argparse
    Dados de saida:
                    - prediction: DataFrame com a predicao do modelo

    Variáveis:
        - A variavel "dataFinal" é referente a data do dia atual, e pode ser alterada para a data que o usuário desejar
        no formato 'str' na ordem yyyy-mm-dd
        - A variavel "dataInicio" é referente a data de inicio do retorno da base e também pode ser alterada para a data que o usuário desejar
        no formato 'str' na ordem yyyy-mm-dd

    """
    logger.info("Carregando pipeline")
    model = load_pipeline(MODEL_PATH)

    logger.info("Carregando os dados")
    
    features = _load_train_data()

    data = _remove_high_correlation(features)
    data = data.drop(['uninstall', 'id_ga'], axis=1)

    logger.info("Fazendo predição")
    
    Result = predict(data, model)

    Result = adicao_da_coluna(features, Result)
    Result = _add_DtRef_TsAtualizacao_dataframe(df = Result, dt_fim = dataFinal)

    logger.info("salvando os dados")
    
    save_predict_data(Result, dt_fim = dataFinal)

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