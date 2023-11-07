# Databricks notebook source
# MAGIC %md
# MAGIC # TESTES DE EXECUÇÃO
# MAGIC   - EXECUTANDO O DATAPREPARATION
# MAGIC   - 

# COMMAND ----------

# # ---------------------------------------------
# !pip install mlflow
# !pip install xgboost
# # ---------------------------------------------

# COMMAND ----------

# ---------------------------------------------
# Configuracoes
DATA_PREPARATION_PATH = ""
DATA_PREPARATION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3_teste"
#DATA_PREPARATION_TABLE_NAME = "databox_marketing_comum.snup_database_modelagem_v3_trainer"
# ----------------------------------------------

# COMMAND ----------

# MAGIC %run "/Users/email-user/crm_modelo_para_uninstall_V1/data_preparation/data_preparation"

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
dataFinal = "2023-09-15"
print(dataInicio)
print(dataFinal)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Preparando os dados

# COMMAND ----------

data_preparation(spark,dt_inicio = dataInicio, dt_fim = dataFinal, mes_ano = mes_ano)

# COMMAND ----------

#data = spark.table("databox_marketing_comum.snup_database_modelagem_v3_teste")
#data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Treinamento do Modelo