# Databricks notebook source
# ---------------------------------------------
!pip install mlflow
!pip install xgboost
# ---------------------------------------------

# COMMAND ----------

# ---------------------------------------------
# Configuracoes
DATA_PREPARATION_PATH = ""
DATA_PREPARATION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3"
#DATA_PREPARATION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3_teste"
#DATA_PREPARATION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3_trainer"
# ----------------------------------------------

# COMMAND ----------

# ---------------------------------------------
EXPERIMENT_ID = 0
MODEL_NAME = 'teste_xgboost_v0'
# ---------------------------------------------

# COMMAND ----------

# MAGIC %run "/Users/email-user/crm_modelo_para_uninstall_V1/trainer/trainer_model"

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
dataFinal = "2023-09-02"
print(dataInicio)
print(dataFinal)

# COMMAND ----------

# ---------------------------------------------
dataInicio = dataInicio
dataFinal = dataFinal
# ---------------------------------------------

# COMMAND ----------

# ---------------------------------------------
# Base para modelagem
database = spark.table(DATA_PREPARATION_TABLE_NAME).toPandas()
# ---------------------------------------------

# COMMAND ----------

# ---------------------------------------------
# Executando o Train
train(dataInicio = dataInicio, dataFinal = dataFinal, experiment_id=EXPERIMENT_ID)
# ---------------------------------------------

# COMMAND ----------

