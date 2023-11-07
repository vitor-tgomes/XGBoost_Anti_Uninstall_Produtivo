# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Pegar ID para ser utilizado
# MAGIC
# MAGIC 1. https://adb-22844899239122#################################################
# MAGIC 2. Clica em `Run Name` nome do experimento
# MAGIC 3. Pega o valor em `Make Predictions`
# MAGIC

# COMMAND ----------

# ---------------------------------------------
MODEL_NAME = 'teste_xgboost_v0'
# ---------------------------------------------
# Carregue o modelo XGBoost treinado a partir do MLflow
model_uri = f'runs:/e1a0e7da9e0a4021a5205387ec306b59/{MODEL_NAME}'


# COMMAND ----------

# MAGIC %run "/Users/email-user/crm_modelo_para_uninstall_V1/predict/predictor"

# COMMAND ----------

# ---------------------------------------------
# Caminho para arquivo de leitura das predicoes
DATA_PREDICTION_PATH = ""
DATA_PREDICTION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3"


# COMMAND ----------

# ---------------------------------------------
# Caminho para salvar arquivo das predicoes
PREDICTION_RESULT_TABLE_PATH = ""
PREDICTION_RESULT_TABLE_NAME = "databox_marketing_comum.snup_database_prediction_result"


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
# ---------------
from datetime import datetime
from datetime import timedelta
# ------------------------------
# Mes e ano para os dados do appsflyer
mes_ano = "2023-09"
# ------------------------------
# Data Inicio e fim para amostragem
Hoje = datetime.today()
diasAtras = timedelta(2)
dataFinal = Hoje.strftime("%Y-%m-%d"); #Referente ao dia de 'ontem'
dataInicio = Hoje - diasAtras;
dataInicio = dataInicio.strftime("%Y-%m-%d");

dataInicio = "2023-09-01"
dataFinal = "2023-09-10"
print(dataInicio)
print(dataFinal)

# COMMAND ----------

# ---------------------------------------------
# Base para predicao
database = spark.table(DATA_PREDICTION_TABLE_NAME).toPandas()
# ---------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Executando o Predict
# MAGIC

# COMMAND ----------

run(dataInicio, dataFinal, MODEL_PATH = model_uri)

# COMMAND ----------

# ---------------------------------------------
# Verificando Salvamento
database_result = spark.table(PREDICTION_RESULT_TABLE_NAME)
# ---------------------------------------------
database_result.display()

# COMMAND ----------

