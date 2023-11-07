# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ##### Preparação dos dados

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, date_format, min, udf, concat, unix_timestamp, format_string, map_values, substring_index, expr
from pyspark.sql.types import IntegerType, DecimalType, DoubleType, StringType
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame

import pandas as pd
import numpy as np

# from src.contrib.config import DATA_PREPARATION_PATH, DATA_PREPARATION_TABLE_NAME
# from src.contrib.logger import logger
# from src.contrib.utils import clean_table_before_insert

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

def clean_table_before_insert(spark: SparkSession, dt_fim: str, table_name: str) -> None:
    '''
        Está função limpa a tabela conforme particições:
            - AaParticao
            - MmParticao
            - DdParticao
        Baseado no dt_fim
    '''
    
    flg_tb_existe = spark._jsparkSession.catalog().tableExists(table_name.split(".")[0], table_name.split(".")[1])
    if flg_tb_existe:
        spark.sql(
            f"DELETE FROM {table_name} WHERE AaParticao = year('{dt_fim}') AND MmParticao = month('{dt_fim}') AND DdParticao = dayofmonth('{dt_fim}')")

# COMMAND ----------

def replace_outliers(df, coluna):
    '''
    Esta funcao:
        - Retira outliers substituindo os valores extremos pela mediana
    Importante:
        -
    Dados de saida:
        - DataFrame com os valores extremos substituídos pela mediana
    '''

    # Calcular a Mediana
    median_value = df.approxQuantile(coluna, [0.5], 0.05)[0]

    # Calcular os outliers 5% superiores
    upper_bound = df.approxQuantile(coluna, [0.95], 0.001)[0]

    # Substituir os Outliers pela mediana
    return df.withColumn(coluna, F.when(F.col(coluna) > upper_bound, median_value).otherwise(F.col(coluna)))

# COMMAND ----------

def load_data_users (spark: SparkSession) -> DataFrame:
    '''
    Esta funcao:
        - Obtem dados do DataLake: carrega tabela do app_corp.cliente
    Importante:
        - É utilizado apenas CASASBAHIA
        - Foi criado faixa de idade
    Dados de saida:
        - DataFrame obtdido com os dados do app_corp.cliente com idade sendo utilizada por faixa etaria
    '''
    
    data = spark.table("app_corp.cliente")\
        .filter(F.col('origem') == 'SITE')\
        .filter(F.col('bandeira') == 'CASASBAHIA')\
        .select('idcliente', 'sexo', 'datanascimento')\
        .withColumn('idade', F.floor(F.datediff(F.current_date(), F.col("datanascimento")) / 365.25))\
        .withColumn(
            'faixa_etaria', 
            F.when(F.col('idade') <= 20 , 'ATE_20')\
            .when(F.col('idade') <= 30 , '21_A_30')\
            .when(F.col('idade') <= 50 , '31_A_50')\
            .otherwise('+_50')
        )\
        .drop('datanascimento', 'idade')\
        .dropDuplicates()

    return data

# COMMAND ----------

def load_data_ga_amostra (spark: SparkSession) -> DataFrame:
    '''
    Esta funcao:
        - Obtem dados do DataLake: carrega tabela do raw_parceiro.googleanalyticsnavegacaoexplode
    Importante:
        - Foi fixado NR_PARTITION_YEAR >= 2022. Caso necessário, rever o perído de coleta das informações da tabela. 
        - Foram extraídas informações referentes a data máxima (D_REF)
    Dados de saida:
                    - DataFrame obtdido com os dados do crm_book agrupados por CPF/CNPJ. Ou seja, informação única dos clientes
    '''
    
    ga = spark.table('raw_parceiro.googleanalyticsnavegacaoexplode')\
        .filter(F.col('OrigemParticao') == 'APP')\
        .filter(F.col('AaParticao').isin('2023'))\
        .filter(F.col('MmParticao').isin('09'))\
        .filter(F.col('BandeiraParticao') == 'CASASBAHIA')\
        .filter(F.col('device.operatingSystem') == 'Android')\
        .withColumn('date', F.to_date(F.col('date').cast(StringType()), 'yyyyMMdd'))\
        .withColumn('hora',  format_string("%02d", F.col('hits.hour').cast('int')))\
        .withColumn('minuto',  format_string("%02d", F.col('hits.minute').cast('int')))\
        .withColumn('timestamp_hit', F.date_format(F.to_timestamp(F.concat("date","hora", "minuto"),"yyyy-MM-ddHHss"),"yyyy-MM-dd hh:ss"))\
        .withColumn('id_ga', F.col('FullVisitorId'))

    ga_amostra = ga\
        .withColumn('custom', F.explode(F.col("customDimensions")))\
        .filter(F.col('custom.index') == 69)\
        .withColumn('advertising_id', F.col('custom.value'))\
        .drop('custom')\
        .filter(F.col('advertising_id') != '00000000-0000-0000-0000-000000000000')\
        .filter(F.col('advertising_id') != "0000-0000")\
        .select(F.col('FullVisitorId').alias('id_ga'), 'advertising_id')

    return ga,ga_amostra


# COMMAND ----------

def load_data_appsflyer (spark: SparkSession) -> DataFrame:
    '''
    Esta funcao:
        - Obtem dados do DataLake: carrega tabela do crm_book
    Importante:
        - Foi fixado NR_PARTITION_YEAR >= 2022. Caso necessário, rever o perído de coleta das informações da tabela. 
        - Foram extraídas informações referentes a data máxima (D_REF)
    Dados de saida:
                    - DataFrame obtdido com os dados do crm_book agrupados por CPF/CNPJ. Ou seja, informação única dos clientes
    '''
    
    data = spark.table('databox_marketing_restrito.appsflyer_events')\
        .filter(F.col('bandeira_partition') == 'CASASBAHIA')\
        .filter(F.col('device') == 'Android')\
        .filter(F.col('event_name') != 'af_login')\
        .dropDuplicates()
    
    return data

# COMMAND ----------

def _create_appsflyer_definitivo (spark: SparkSession, mes_ano: str) -> DataFrame:
    '''
    Esta funcao:
        - Obtem dados do DataLake: carrega a tabela de tbclientepedidoitem
    Importante:
        - NULL.    
    Dados de saida:
                    - data: DataFrame itempedido com ID dos clientes, id dos produtos, bandeira e hora do pedido.
    '''
    # month('{dt_fim}')
    #                          WHERE AaParticao = year('{dt_fim}')\
    #                          AND MmParticao = month('{dt_fim}') \

    appsflyer = load_data_appsflyer(spark)

    source = ['restricted', 'banner_home_app', 'facebook ads', 'googleadwords_int', 'bytedanceglobal_int', 'shareit_int']

    #instalações appsflyer tratado
    installs = appsflyer\
        .withColumn('y_monthref', F.date_format(F.col('event_time_sm'), 'yyyy-MM'))\
        .filter((F.col("y_monthref") == mes_ano))\
        .filter(F.col('event_name').isin('install','reinstall'))\
        .filter((F.col('advertising_id').isNull() == False) & (F.col('advertising_id') != '00000000-0000-0000-0000-000000000000') & (F.col('advertising_id') != '0000-0000'))\
        .withColumn('atribuicao_canais', F.when(F.col('atribuicao_canais').isin('Organico', 'Install Aquisicao'), F.col('atribuicao_canais')).otherwise('Outros'))\
        .withColumn('atribuicao_canais', F.when(F.col('atribuicao_canais') == 'Install Aquisicao', 'install_aquisicao').otherwise(F.col('atribuicao_canais')))\
        .withColumn('media_source', 
                    F.when(F.col('media_source').isNull() == True, 'na')\
                    .when(F.col('media_source').isin(source), F.col('media_source'))\
                    .otherwise('Outros')
        )\
        .select(
            F.col('advertising_id').alias('advertising_appsflyer'), 'appsflyer_id', F.col('event_time').alias('install_date'), F.col('event_time_sm').alias('install_date_sm'), 
            F.col('atribuicao_canais').alias('install_grouping'), F.col('channel').alias('install_channel'), F.col('media_source').alias('install_source'))\
        .withColumn('install_date', F.to_timestamp(F.col('install_date')))\
        .withColumn('install_date_sm', F.to_date(F.col('install_date_sm')))\
        .withColumn('install_hour', F.hour(F.col('install_date')))\
        .withColumn('install_manha', F.when((F.col('install_hour') >= 6) & (F.col('install_hour') < 12), 1).otherwise(0))\
        .withColumn('install_tarde', F.when((F.col('install_hour') >= 12) & (F.col('install_hour') < 19), 1).otherwise(0))\
        .withColumn('install_noite', F.when((F.col('install_hour') >= 19) & (F.col('install_hour') <= 23), 1).otherwise(0))\
        .withColumn('install_madrugada', F.when((F.col('install_hour') >= 0) & (F.col('install_hour') < 6), 1).otherwise(0))\
        .withColumn('install', F.lit('1'))\
        .dropDuplicates()\
        .withColumn('qtd_installs', F.count('install').over(Window().partitionBy('advertising_appsflyer')))\
        .filter(F.col('qtd_installs') == 1)\
        .drop('qtd_installs', 'install', 'install_hour')

    #Auxiliar installs antigos
    install_antigo = appsflyer\
        .filter(F.col('event_name').isin('install','reinstall'))\
        .filter((F.col('advertising_id').isNull() == False) & (F.col('advertising_id') != '00000000-0000-0000-0000-000000000000') & (F.col('advertising_id') != '0000-0000'))\
        .select(F.col('advertising_id').alias('advertising_antigo'), F.col('event_time').alias('install_date_antigo'))\
        .withColumn('install_date_antigo',  F.to_timestamp(F.col('install_date_antigo')))\
        .withColumn('install_antigo', F.lit(1))\
        .dropDuplicates()

    uninstall = appsflyer\
    .filter(F.col('event_name').isin('uninstall'))\
    .filter((F.col('advertising_id').isNull() == False) & (F.col('advertising_id') != '00000000-0000-0000-0000-000000000000') & (F.col('advertising_id') != '0000-0000'))\
    .select(F.col('appsflyer_id').alias('appsflyer_id_un'), F.col('event_time').alias('uninstall_date'))\
    .withColumn('uninstall_date',  F.to_timestamp(F.col('uninstall_date')))\
    .withColumn('uninstall', F.lit(1))\
    .dropDuplicates()

    cond = (
    (installs['advertising_appsflyer'] == install_antigo['advertising_antigo']) &\
    (installs['install_date'] > install_antigo['install_date_antigo'])
    )

    cond_2 = (
        (installs['appsflyer_id'] == uninstall['appsflyer_id_un']) &\
        (installs['install_date'] < uninstall['uninstall_date'])&\
        (F.date_add(installs['install_date'], 8) >= uninstall['uninstall_date'])
    )


    data = installs\
        .join(install_antigo, on=cond, how='left')\
        .join(uninstall, on=cond_2, how='left')\
        .fillna(0, subset=['install_antigo', 'uninstall'])\
        .groupBy('advertising_appsflyer','appsflyer_id','install_date','install_date_sm', 'install_manha', 'install_tarde', 'install_noite', 'install_madrugada','install_grouping','install_channel','install_source', 'uninstall')\
        .agg(
            F.sum('install_antigo').alias('total_installs_antigos'),
            F.max('install_date_antigo').alias('last_install_antigo')
        )\
        .withColumn('recencia_last_install', F.datediff(F.col('install_date_sm'), F.col('last_install_antigo')))\
        .fillna(-1, subset=['recencia_last_install'])\
        .drop('last_install_antigo', 'install_date', 'install_antigo')

    return data 

# COMMAND ----------

def _crete_ga_hit_info (spark: SparkSession, ga_bruto: DataFrame, dados_cli: DataFrame) -> DataFrame:
    '''
    Esta funcao:
        - Obtem dados do DataLake: carrega a tabela de tbclientepedidoitem
    Importante:
        - NULL.    
    Dados de saida:
                    - data: DataFrame itempedido com ID dos clientes, id dos produtos, bandeira e hora do pedido.
    '''
    #--------------------------------------------
    sudeste = ['State of Sao Paulo', 'State of Rio de Janeiro', 'State of Minas Gerais', 'State of Espirito Santo']
    sul = ['State of Parana', 'State of Rio Grande do Sul', 'State of Santa Catarina']
    norte = ['State of Acre',  'State of Amapa', 'State of Amazonas', 'State of Para', 'State of Rondonia', 'State of Roraima', 'State of Tocantins']
    c_oeste = ['Federal District', 'State of Mato Grosso', 'State of Goias', 'State of Mato Grosso do Sul']
    #--------------------------------------------
    template_list = ['busca','produto','home','conta','colecao']
    event_v1 = [
        'produto_buscou_duvida', 'produto_navegou', 'acesso_clicou', 'produto_calculou', 'produto_clicou',
        'carrinho_calculou', 'produto_favoritou', 'optin-push-sys_clicou', 'enhanced_ecommerce_add_to_cart',
        'enhanced_ecommerce_remove_from_cart','enhanced_ecommerce_begin_checkout', 'enhanced_ecommerce_checkout_progress', 'compra finalizada_clicou'
    ]
    event_v2 = [
        'acesso_clicou_clicou talvez mais tarde', 'produto_clicou_midia avaliacoes', 'optin-push-sys_clicou_permitiu',
        'erro_exibiu_desculpe! no momento este produto nao pode ser entregue na regiao informada. (id-1200)',
        'acesso_clicou_senha', 'acesso_clicou_entrar com facebook', 'acesso_clicou_cadastro concluido', 'acesso_clicou_ativou biometria'
    ]
    core_produtos = [
    'moveis','celulares e telefones','eletrodomesticos','portateis','tv e video'
    ]
    calda = [
        'utilidades domesticas','esporte e lazer','automotivo','ar e ventilacao','calcados','informatica','bebes','ferramentas',
        'audio','games','saude e beleza','acessorios e inovacoes','perfumaria e cosmeticos'
    ]
    #--------------------------------------------
    ga_hit_info = ga_bruto\
        .withColumn('session_id', F.concat(F.col('id_ga'), F.col('visitId')))\
        .withColumn('visitStartTime', F.to_timestamp(F.min('timestamp_hit').over(Window().partitionBy('session_id'))))\
        .withColumn('visitEndTime', F.to_timestamp(F.max('timestamp_hit').over(Window().partitionBy('session_id'))))\
        .withColumn('SessionVisitTime_minutes', F.round((F.col('visitEndTime').cast("long") - (F.col('visitStartTime').cast("long")))/60, 2))\
        .withColumn('custom_hit', F.explode(F.col('hits.customDimensions')))\
        .withColumn('hit_number', F.col('hits.HitNumber'))\
        .withColumn('hit_time', F.col('hits.time'))\
        .withColumn('hit_rank', F.dense_rank().over(Window().partitionBy('session_id').orderBy(F.col('hits.hitNumber'))))\
        .withColumn('template_screen', F.when((F.col('custom_hit.index') == 5) & (F.col('custom_hit.value') != ""), F.col('custom_hit.value')))\
        .withColumn('depart_pageview', F.when((F.col('custom_hit.index') == 6) & (F.col('custom_hit.value') != ""), F.col('custom_hit.value')))\
        .withColumn('event_v1', F.concat_ws('_', F.col('hits.eventInfo.eventCategory'), F.col('hits.eventInfo.eventAction')))\
        .withColumn('event_v2', F.concat_ws('_', F.col('hits.eventInfo.eventCategory'), F.col('hits.eventInfo.eventAction'), F.col('hits.eventInfo.eventLabel')))\
        .withColumn(
            'screen_view',
            F.when(F.col('template_screen').isin(template_list), F.col('template_screen'))
        )\
        .withColumn(
            'produto_view',
            F.when(F.col('depart_pageview').isin(core_produtos), 'core')\
            .when(F.col('depart_pageview').isin(calda) == False, 'calda')
        )\
        .withColumn(
            'action_event_v1',
            F.when(F.col('event_v1').isin(event_v1), F.col('event_v1'))
        )\
        .withColumn(
            'action_event_v2',
            F.when(F.col('event_v2').isin(event_v2), F.col('event_v2'))
        )\
        .withColumn(
            'regioes',
            F.when(F.col('geoNetwork.region').isin(sudeste), 'sudeste')\
            .when(F.col('geoNetwork.region').isin(sul), 'sul')\
            .when(F.col('geoNetwork.region').isin(norte), 'norte')\
            .when(F.col('geoNetwork.region').isin(c_oeste), 'centro_oeste')\
            .when(F.col('geoNetwork.region').like('%State of'), 'nordeste')
        )\
        .join(dados_cli, on=['idcliente'], how='left')\
        .select('id_ga', 'sexo', 'faixa_etaria', 'session_id', 'totals.hits', 'totals.screenviews', 'totals.UniqueScreenviews', 'totals.sessionQualityDim',  'visitNumber', 'channelGrouping',  'SessionVisitTime_minutes', 'regioes',
                'install_manha', 'install_tarde','install_noite','install_madrugada','install_grouping','install_channel','total_installs_antigos', 'recencia_last_install', 'trafficSource.isTrueDirect',
                'screen_view', 'produto_view', 'action_event_v1', 'action_event_v2', 'uninstall')\
        .dropDuplicates()

    return ga_hit_info

# COMMAND ----------

def _load_data_join_appsflyer_ga_usuarios (spark: SparkSession, dt_inicio: str, dt_fim: str, mes_ano: str) -> DataFrame: 

    '''
    Esta funcao:
        - Obtem dados do DataLake: Aplica um join nas tabelas de itempedido, audioperform, crm_book, google_trends
    Importante:
        -
    Dados de saida:
                    - data: Retorna um dataframe com as informações das quatro tabelas 
                    - variáveis de crmbook: NR_CPF_CNPJ,DT_REF,VR_GMV_VIDA_ON, VR_TICKET_MEDIO,DS_GENERO,DS_ESTADOCIVIL,NR_IDADE
                    - variáveis de audioperform: cpfcnpj,DtAtualizacao,CdUF,NmMelhorBandeira
                    - variáveis de trends: SKU, dsbandeira, dsitemsite, MelhorEstado, InteresseGeral
                    - variáveis de tbclienteitempedido: DsSingleId, NrCpfCliente, NrItemSite, DsBandeira, DhPedido
                                    
     - Dados de entrada:
         - dataframe1: compras 
         - dataframe2: trends
         - dataframe3: crm
         - dataframe4: audio
    '''
    
    #--------------------------------------------
    #mes_ano = '2023-09'
    dataframe1 = _create_appsflyer_definitivo(spark, mes_ano)
    ga,ga_amostra = load_data_ga_amostra(spark)   
    dataframe3 = load_data_users(spark)
    #--------------------------------------------
    appsflyer_amostra = dataframe1\
        .filter((F.col('install_date_sm') >= dt_inicio) & (F.col('install_date_sm') <= dt_fim)) 
    #--------------------------------------------
    ga_amostra_appsflyer = ga_amostra\
        .join(appsflyer_amostra.withColumnRenamed('advertising_appsflyer', 'advertising_id'), on=['advertising_id'], how='inner')\
        .dropDuplicates()
    
    ga_definitivo = ga\
    .join(ga_amostra_appsflyer, on=['id_ga'], how='inner')\
    .filter(F.col('date') >= F.col('install_date_sm'))\
    .filter(F.col('date') <= F.date_add(F.col('install_date_sm'), 1))
    #--------------------------------------------
    ga_bruto = ga_definitivo\
        .withColumn("logado",F.expr("transform(filter(customDimensions, x -> x.index = 4), x -> x.value)[0]"))\
        .withColumn("logado", F.when(F.col("logado") == "logado", 1))\
        .withColumn("idcliente",F.expr("transform(filter(customDimensions, x -> x.index = 68), x -> x.value)[0]"))
    #--------------------------------------------
    ga_analise = ga_bruto\
        .withColumn('session_id', F.concat(F.col('id_ga'), F.col('visitId')))\
        .withColumn('visitStartTime', F.to_timestamp(F.min('timestamp_hit').over(Window().partitionBy('session_id'))))\
        .withColumn('visitEndTime', F.to_timestamp(F.max('timestamp_hit').over(Window().partitionBy('session_id'))))\
        .withColumn('SessionVisitTime_minutes', F.round((F.col('visitEndTime').cast("long") - (F.col('visitStartTime').cast("long")))/60, 2))\
        .withColumn('custom_hit', F.explode(F.col('hits.customDimensions')))\
        .withColumn('hit_number', F.col('hits.HitNumber'))\
        .withColumn('hit_time', F.col('hits.time'))\
        .withColumn('hit_rank', F.dense_rank().over(Window().partitionBy('session_id').orderBy(F.col('hits.hitNumber'))))\
        .withColumn('template_screen', F.when((F.col('custom_hit.index') == 5) & (F.col('custom_hit.value') != ""), F.col('custom_hit.value')))\
        .withColumn('depart_pageview', F.when((F.col('custom_hit.index') == 6) & (F.col('custom_hit.value') != ""), F.col('custom_hit.value')))\
        .withColumn('event_v1', F.concat_ws('_', F.col('hits.eventInfo.eventCategory'), F.col('hits.eventInfo.eventAction')))\
        .withColumn('event_v2', F.concat_ws('_', F.col('hits.eventInfo.eventCategory'), F.col('hits.eventInfo.eventAction'), F.col('hits.eventInfo.eventLabel')))
    #--------------------------------------------
    ga_hit_info = _crete_ga_hit_info(spark, ga_bruto, dataframe3)
    #--------------------------------------------
    final_tratado = ga_hit_info\
        .withColumn('rank_install', F.dense_rank().over(Window().partitionBy('id_ga').orderBy(F.col('visitNumber'))))\
        .withColumn(
            'channelGrouping_latencia0',
            F.when(F.col('rank_install') == 1, F.concat(F.lit('install : '), F.col('install_grouping')))\
            .when(F.col('isTrueDirect') == 'true', F.lit('Direto'))\
            .otherwise(F.col('channelGrouping'))
        )\
        .drop('isTrueDirect', 'channelGrouping', 'install_grouping', 'visitNumber', 'rank_install')
  
    return final_tratado 

# COMMAND ----------

def _load_data_for_model (spark: SparkSession, dt_inicio: str, dt_fim: str, mes_ano: str) -> DataFrame:
    
    '''
    Esta funcao:
        - Obtem dados do DataLake: Aplica um join nas tabelas de itempedido, audioperform, crm_book, google_trends
    Importante:
        -
    Dados de saida:
                    - data: Retorna um dataframe com as informações das quatro tabelas 
                    - variáveis de crmbook: NR_CPF_CNPJ,DT_REF,VR_GMV_VIDA_ON, VR_TICKET_MEDIO,DS_GENERO,DS_ESTADOCIVIL,NR_IDADE
                    - variáveis de audioperform: cpfcnpj,DtAtualizacao,CdUF,NmMelhorBandeira
                    - variáveis de trends: SKU, dsbandeira, dsitemsite, MelhorEstado, InteresseGeral
                    - variáveis de tbclienteitempedido: DsSingleId, NrCpfCliente, NrItemSite, DsBandeira, DhPedido
                                    
     - Dados de entrada:
         - dataframe1: compras 
         - dataframe2: trends
         - dataframe3: crm
         - dataframe4: audio
    '''
    logger.info("Inicio - ga - appsflyes - usuarios")
    dados = _load_data_join_appsflyer_ga_usuarios(spark,dt_inicio, dt_fim, mes_ano)
    dados.persist()
    logger.info("Fim - ga - appsflyes - usuarios")

    logger.info("Inicio - amostra")
    amostra = dados\
        .withColumn('action_event_v2',
                    F.when(F.col('action_event_v2') == 'erro_exibiu_desculpe! no momento este produto nao pode ser entregue na regiao informada. (id-1200)', 'erro_regiao_nao_entregue')\
                    .when(F.col('action_event_v2').isin('acesso_clicou_senha', 'acesso_clicou_entrar com facebook', 'acesso_clicou_cadastro concluido', 'acesso_clicou_ativou biometria'), 'acesso_login_realizado')\
                    .otherwise(F.col('action_event_v2')))\
                    .fillna('sem_info', subset=['sexo', 'faixa_etaria'])\
                    .withColumn('sexo', 
                                F.when(F.col('sexo').isin('M', 'm'), 'M')\
                                .when(F.col('sexo').isin('F', 'f'), 'F')\
                                .otherwise('sem_info')
                    )
    amostra.persist()
    logger.info("Fim - amostra - (finalizou aqui com 39 min)")

    logger.info("Inicio - user hit - sessoes")
    amostra = amostra.repartition(100)  # Substitua 100 pelo número desejado de partições.
    user_hitinfo = amostra\
        .groupBy('id_ga')\
        .agg(
            F.collect_set('screen_view').alias('screen_view'),
            F.collect_set('produto_view').alias('produto_view'),
            F.collect_set('action_event_v1').alias('action_event_v1'),
            F.collect_set('action_event_v2').alias('action_event_v2')
        )
    logger.info("Fim - user_hitinfo- (finalizou aqui com 56 min)")

    user_sessoes = amostra\
        .drop('screen_view', 'produto_view', 'action_event_v1', 'action_event_v2')\
        .dropDuplicates()

    user_sessoes = user_sessoes\
        .withColumn(
            'channelGrouping_latencia0',
            F.when(F.col('channelGrouping_latencia0').like('install :%'), F.col('channelGrouping_latencia0'))\
            .when(F.col('channelGrouping_latencia0').isin('Direto', 'CRM - Marketing Direto', 'SEM - Shopping', 'Remarketing'), F.col('channelGrouping_latencia0'))\
            .otherwise('outros_canais')
        )
    user_sessoes.display()
    logger.info("Fim - user_sessoes - ")

    ### Foi comentado abaixo para ir metrificando o tempo até os passos não comentados
    # user_sessoes_outliers = user_sessoes
    # colunas_outliers = ['hits', 'screenviews', 'UniqueScreenviews', 'sessionQualityDim', 'SessionVisitTime_minutes']

    # for coluna in colunas_outliers:
    #     user_sessoes_outliers = replace_outliers(user_sessoes_outliers, coluna)

    logger.info("Fim - user hit - sessoes (finalizou aqui com 6.51 horas)")

    
    logger.info("Inicio - user_sessoes_vfinal")
    # user_sessoes_vfinal = user_sessoes_outliers\
    #     .groupBy('id_ga', 'faixa_etaria', 'sexo', 'uninstall')\
    #     .agg(
    #         F.countDistinct('session_id').alias('sessions'),
    #         F.round(F.avg('hits'),2).alias('media_hits'),
    #         F.round(F.avg('screenviews'),2).alias('media_screenviews'),
    #         F.round(F.avg('UniqueScreenviews'),2).alias('media_uniquescreens'),
    #         F.round(F.avg('sessionQualityDim'),2).alias('media_qualitysession'),
    #         F.round(F.avg('SessionVisitTime_minutes'),2).alias('media_timesession'),
    #         F.sum('SessionVisitTime_minutes').alias('soma_timesession'),
    #         F.max('install_manha').alias('install_manha'),
    #         F.max('install_tarde').alias('install_tarde'),
    #         F.max('install_noite').alias('install_noite'),
    #         F.max('install_madrugada').alias('install_madrugada'),
    #         F.max('total_installs_antigos').alias('totals_installs_antigos'),
    #         F.max('recencia_last_install').alias('recencia_last_install'),
    #         F.max('regioes').alias('regioes'),
    #         F.collect_set('install_channel').alias('install_channel'),
    #         F.collect_set('channelGrouping_latencia0').alias('canais')
    #         )\
    #     .join(user_hitinfo, on=['id_ga'], how='inner')\
    #     .withColumn('install_channel', F.explode_outer(F.col('install_channel')))\
    #     .withColumn('canais', F.explode_outer(F.col('canais')))\
    #     .withColumn('screen_view', F.explode_outer(F.col('screen_view')))\
    #     .withColumn('produto_view', F.explode_outer(F.col('produto_view')))\
    #     .withColumn('action_event_v1', F.explode_outer(F.col('action_event_v1')))\
    #     .withColumn('action_event_v2', F.explode_outer(F.col('action_event_v2')))
    logger.info("Fim - user_sessoes_vfinal")

    logger.info("Inicio - parte pandas")
    # # Parte Pandas
    # user_sessoes_pandas = user_sessoes_vfinal.toPandas()
    # user_sessoes_dummi = pd.get_dummies(user_sessoes_pandas, columns=['faixa_etaria', 'sexo', 'screen_view', 'produto_view', 'action_event_v1', 'action_event_v2', 'regioes', 'install_channel', 'canais'])

    # colunas_dummi_first = [coluna for coluna in user_sessoes_dummi.columns if ("screen_view" in coluna) or ("produto_view" in coluna) or ("action_event" in coluna) or ("regioes" in coluna) or ("install_channel" in coluna) 
    #                 or ("canais" in coluna) or ("faixa_etaria" in coluna) or ("sexo" in coluna)]

    # colunas_outras = [coluna for coluna in user_sessoes_dummi.columns if (coluna not in colunas_dummi_first)]

    # aggregations = {col: 'max' for col in colunas_dummi_first}

    # user_sessoes_dummi_final = user_sessoes_dummi\
    #     .groupby(colunas_outras)\
    #     .agg(
    #         aggregations
    #     )\
    #     .reset_index()
    logger.info("Fim - parte pandas")

    logger.info("Inicio - Create Spark DataFrame final")
    # base_modelagem = spark.createDataFrame(user_sessoes_dummi_final)
    logger.info("Fim - Create Spark DataFrame final")

    # return base_modelagem



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

def data_preparation_run(spark: SparkSession, dt_inicio: str, dt_fim: str, mes_ano:str) -> DataFrame:
    
    """
    Esta funcao:
        - Através dos dados obtidos do _load_data_join_appsflyer_ga_usuarios()
        faz as transformações necessárias para utilização do modelo.

    Importante:
        - 

    Dados de Entrada:
                    - dt_inicio: String que representa a data inicial para retorno da base
                    - dt_fim: String que representa a data final para retorno da base
    Dados de saida:
                    - df: DataFrame que possui as seguintes variaveis
                        - 
    """

    df = _load_data_for_model(spark,dt_inicio, dt_fim, mes_ano)
    #df = _add_DtRef_TsAtualizacao_dataframe(df = df, dt_fim = dt_fim)
    
    clean_table_before_insert(spark, dt_fim, DATA_PREPARATION_TABLE_NAME)

    # df.write.partitionBy("AaParticao",\
    #                      "MmParticao",\
    #                      "DdParticao")\
    #                     .format("delta")\
    #                     .option("mergeSchema", "true")\
    #                     .mode("append")\
    #                     .saveAsTable(DATA_PREPARATION_TABLE_NAME)
    #                     #.saveAsTable(DATA_PREPARATION_TABLE_NAME, \
    #                     #             path = DATA_PREPARATION_PATH)

    return df

# COMMAND ----------

def data_preparation(spark: SparkSession, dt_inicio: str, dt_fim: str, mes_ano:str) ->  DataFrame:
    
    """
    Esta funcao:
        - Verifica se as tabelas requeridas já existem.


    Dados de Entrada:
                    - dt_inicio: String que representa a data inicial para retorno da base
                    - dt_fim: String que representa a daa final para retorno da base
    Dados de saida:
                    - DataFrame: Tabela com os dados para execução do algoritmo
       
    """ 
    
    flg_tb_existe = spark._jsparkSession.catalog().tableExists(DATA_PREPARATION_TABLE_NAME.split(".")[0],\
                                                               DATA_PREPARATION_TABLE_NAME.split(".")[1])
    
    if flg_tb_existe:
        features = spark.sql(f"SELECT * FROM {DATA_PREPARATION_TABLE_NAME}\
                             WHERE AaParticao = year('{dt_fim}')\
                             AND MmParticao = month('{dt_fim}') \
                             AND DdParticao = dayofmonth('{dt_fim}')")
        
        if features.count() <= 0:
            features = data_preparation_run(spark, dt_inicio = dt_inicio, dt_fim = dt_fim, mes_ano = mes_ano)
    else:
        features = data_preparation_run(spark, dt_inicio = dt_inicio, dt_fim = dt_fim, mes_ano = mes_ano)
        
    return features

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