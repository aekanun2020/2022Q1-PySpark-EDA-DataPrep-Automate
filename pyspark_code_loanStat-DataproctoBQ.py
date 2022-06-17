#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark.sql import SparkSession
sc = pyspark.SparkContext()
spark = SparkSession(sc)


# In[1]:


import time as t


# In[2]:


start_time = t.time()


# # 1. Data Understanding

# ## Collecting the current data from the LendingClub's Website.

# In[3]:


raw_df = spark.read.format('csv').option('header','true').option('mode','DROPMALFORMED').load('gs://aekanun-21jun2022/raw_data/LoanStats_web.csv')


# In[4]:


raw201617_df = raw_df


# In[5]:


rawweb_df = raw201617_df.select('id',
 'member_id',
 'loan_amnt',
 'funded_amnt',
 'funded_amnt_inv',
 'term',
 'int_rate',
 'installment',
 'grade',
 'sub_grade',
 'emp_title',
 'emp_length',
 'home_ownership',
 'annual_inc',
 'verification_status',
 'issue_d',
 'loan_status',
 'pymnt_plan',
 'url',
 'desc',
 'purpose',
 'title',
 'zip_code',
 'addr_state',
 'dti',
 'delinq_2yrs',
 'earliest_cr_line',
 'inq_last_6mths',
 'mths_since_last_delinq',
 'mths_since_last_record',
 'open_acc',
 'pub_rec',
 'revol_bal',
 'revol_util',
 'total_acc',
 'initial_list_status',
 'out_prncp',
 'out_prncp_inv',
 'total_pymnt',
 'total_pymnt_inv',
 'total_rec_prncp',
 'total_rec_int',
 'total_rec_late_fee',
 'recoveries',
 'collection_recovery_fee',
 'last_pymnt_d',
 'last_pymnt_amnt',
 'next_pymnt_d',
 'last_credit_pull_d',
 'collections_12_mths_ex_med',
 'mths_since_last_major_derog',
 'policy_code',
 'application_type',
 'annual_inc_joint',
 'dti_joint',
 'verification_status_joint',
 'acc_now_delinq',
 'tot_coll_amt',
 'tot_cur_bal',
 'open_acc_6m',
 #'open_il_6m',
 'open_il_12m',
 'open_il_24m',
 'mths_since_rcnt_il',
 'total_bal_il',
 'il_util',
 'open_rv_12m',
 'open_rv_24m',
 'max_bal_bc',
 'all_util',
 'total_rev_hi_lim',
 'inq_fi',
 'total_cu_tl',
 'inq_last_12m')


# In[6]:


raw_df = rawweb_df


# In[7]:


df_colfam5 = raw_df.select("loan_amnt","term","int_rate","installment","grade","emp_length",                           "home_ownership","annual_inc","verification_status","loan_status",                           "purpose","addr_state","dti","delinq_2yrs","earliest_cr_line",                           "inq_last_6mths","open_acc","pub_rec","revol_bal","revol_util","total_acc",                           "last_credit_pull_d")


# # 2. Data Preparation

# Divide this process into 2 parts. The first is a business oriented preparation that turn many business rules to be programming's logics. Its result benefits many tasks related to the Business Intelligence and other descriptive analytics. The second one is a data science oriented preparation that turn many requirements of data science to be programming's logics. Its result benefits many tasks related to the Predictive analytics.

# Asumption: Business oriented preparation: No missing values, ONLY month, Correct data types.

# ## Data Cleansing: Remove missing values

# In[8]:


df_no_missing = df_colfam5.dropna(how='any')


# In[9]:


df_no_missing_fitmem = df_no_missing.repartition(60)


# In[10]:


df_no_missing_cached = df_no_missing_fitmem.cache()


# In[11]:


df_no_missing_cached.registerTempTable("df")


# ## Data Transformation: Remove Sign of Percent and Extract Month.

# In[12]:


from pyspark.sql.functions import udf
from pyspark.sql.types import *


# ### Remove Sign of Percent

# In[13]:


def f_removepercentsign(origin):
    return origin.rstrip('%')


# In[14]:


removepercentsign = udf(lambda x: f_removepercentsign(x),StringType())


# ### Extract Month

# In[15]:


def f_extractmonth(origin):
    return origin.split('-')[0]


# In[16]:


extractmonth = udf(lambda x: f_extractmonth(x),StringType())


# In[17]:


extractterm = udf(lambda x: x.replace('months',''),StringType())


# In[18]:


from pyspark.sql.functions import col


# In[19]:


dfWithCrunch = df_no_missing_cached.withColumn('revol_util',removepercentsign(col('revol_util')).cast(DoubleType())).withColumn('int_rate',removepercentsign(col('int_rate')).cast(DoubleType())).withColumn('earliest_cr_line',extractmonth(col('earliest_cr_line')).cast(StringType())).withColumn('last_credit_pull_d',extractmonth(col('last_credit_pull_d')).cast(StringType())).withColumn('dti',col('dti').cast(DoubleType())).withColumn('loan_amnt',col('loan_amnt').cast(DoubleType())).withColumn('revol_bal',col('revol_bal').cast(DoubleType())).withColumn('term',extractterm(col('term')).cast(DoubleType())).withColumn('installment',col('installment').cast(DoubleType())).withColumn('open_acc',col('open_acc').cast(DoubleType())).withColumn('total_acc',col('total_acc').cast(DoubleType())).withColumn('pub_rec',col('pub_rec').cast(DoubleType())).withColumn('annual_inc',col('annual_inc').cast(DoubleType()))


# In[ ]:





# In[20]:


rawhive_df = dfWithCrunch.repartition(60).cache()
#rawhive_df.registerTempTable("crunched_data")


# ### Data Transformation: Normalization of "annual_inc" and "loan_amnt"

# In[21]:


from pyspark.sql.functions import *

max_annual_inc = rawhive_df.select(max('annual_inc')).collect()[0][0]

min_annual_inc = rawhive_df.select(min('annual_inc')).collect()[0][0]

#sqlContext.udf.register("t_annual_inc", lambda x: ((x-min_annual_inc)/(max_annual_inc-min_annual_inc)))


# In[22]:


def t_annual_inc(origin):
    return ((origin-min_annual_inc)/(max_annual_inc-min_annual_inc))


# In[23]:


normalized_annual_inc = udf(lambda x: t_annual_inc(x),DoubleType())


# In[24]:


max_loan_amnt = rawhive_df.select(max('loan_amnt')).collect()[0][0]

min_loan_amnt = rawhive_df.select(min('loan_amnt')).collect()[0][0]

#sqlContext.udf.register("t_loan_amnt", lambda x: ((x-min_loan_amnt)/(max_loan_amnt-min_loan_amnt)))


# In[25]:


def t_loan_amnt(origin):
    return ((origin-min_loan_amnt)/(max_loan_amnt-min_loan_amnt))


# In[26]:


normalized_loan_amnt = udf(lambda x: t_loan_amnt(x),DoubleType())


# In[27]:


normalized_df = rawhive_df.withColumn('loan_amnt',normalized_loan_amnt(col('loan_amnt'))).withColumn('annual_inc',normalized_annual_inc(col('annual_inc')))


# ### Number of data rows that are only "Fully Paid" and "Charged Off"

# In[28]:


from pyspark.sql.functions import col


# In[29]:


normalized_filtered_df = normalized_df.filter(col('loan_status') == 'Fully Paid').union(normalized_df.filter(col('loan_status') == 'Charged Off'))


# In[30]:


data = normalized_filtered_df.repartition(60).cache()


# ### Drop Null

# In[31]:


data_no_missing_df = data.dropna(how='any')


# In[5]:


#data_no_missing_df.select("loan_amnt","term","int_rate","dti","loan_status").show()


# In[6]:


# Use the Google Cloud Storage bucket for temporary BigQuery export data used
# by the InputFormat. This assumes the Google Cloud Storage connector for
# Hadoop is configured.
bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')
input_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_input'.format(bucket)


# In[7]:


output_dataset = 'airflow_workshop'
output_table = 'loanstatus_output'


# In[8]:


# Stage data formatted as newline-delimited JSON in Google Cloud Storage.
output_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_output'.format(bucket)
#partitions = range(word_counts.getNumPartitions())
output_files = output_directory + '/part-*'


# In[11]:


data_no_missing_df.select("loan_amnt","term","int_rate","dti","loan_status").write.mode('overwrite').format('csv').save(output_directory)


# In[21]:


#! bq rm -f bde4_dataset


# In[18]:


#! bq mk bde4_dataset


# In[16]:


import subprocess
# Shell out to bq CLI to perform BigQuery import.
subprocess.check_call(
    'bq load --source_format=CSV  '
    '--replace '
    '--autodetect '
    '{dataset}.{table} {files} '
    'loan_amnt:FLOAT,term:FLOAT,int_rate:FLOAT,dti:FLOAT,loan_status:STRING'
    .format(
        dataset=output_dataset, table=output_table, files=output_files
    ).split())


# In[14]:


#output_files


# In[15]:


#!gsutil ls gs://aekanunlab/hadoop/tmp/bigquery/pyspark_output/part-*


# In[ ]:




