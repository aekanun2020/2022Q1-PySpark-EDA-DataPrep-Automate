  #!/usr/bin/env python
# coding: utf-8


import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = pyspark.SparkContext()


# #### Instruction: Before run this code, please create table using cloud shell with "bq mk nasa_dataset".

# ## Data Collection

# #### Download raw data from a data source, and create a RDD from them

# In[1]:


raw_rdd = sc.textFile('gs://aekanuntest/data/nasa.dat')


# ## Data Parsing

# #### Make pattern matching for extracting some information from raw data.

# In[2]:


import time
import datetime
import re
from pyspark.sql import Row

APACHE_ACCESS_LOG_PATTERN = '(\S*) - - \[(\d{2})\/(\S*)\/(\d{4}):(\d{2}):(\d{2}):(\d{2}) (\S*)\]'


# In[3]:


def bejoindate(year,month,date):
    s = '-'
    seq = (year,month,date)
    return s.join(seq)

def bejointime(hour,minute,second):
    s = ':'
    seq = (hour,minute,second)
    return s.join(seq)

def bejoindatetime(date_name,time_name):
    s = ' '
    seq = (date_name,time_name)
    return s.join(seq)

def totimestamp(dt):
    return time.mktime(datetime.datetime.    strptime(dt, "%Y-%b-%d %H:%M:%S").timetuple())


# In[4]:


def parse_apache_log_line(logline):
    pattern = re.compile(APACHE_ACCESS_LOG_PATTERN)
    result = pattern.match(logline)
    if result is None:
        return Row(
        datetime_stamp = None,
        ip_addr = None,
        day_of_month = None,
        month = None,
        year = None,
        hour = None,
        minute = None,
        second = None,
        timezone = None
        )
    return Row(
        #นำวันเดือนปีถูกแยกมาก่อนหน้านี้ กลับมา Join กันใหม่ใน Format ที่เหมาะสม
        datetime_stamp = totimestamp(bejoindatetime(bejoindate(result.group(4).zfill(2),\
                                                               result.group(3),result.group(2).zfill(2)),\
                                                    bejointime(result.group(5),result.group(6),result.group(7)))),
        ip_addr = result.group(1),
        day_of_month = result.group(2),
        month = result.group(3),
        year = result.group(4),
        hour = result.group(5),
        minute = result.group(6),
        second = result.group(7),
        timezone = result.group(8)
        )


# In[5]:


parsed_rdd = raw_rdd.map(parse_apache_log_line)


# #### Create a schema for the parsed data, and make data cleansing, and store data with their schema into the DataFrame.

# In[6]:


raw_df = parsed_rdd.toDF()


# In[7]:


from pyspark.sql.types import IntegerType, DecimalType, TimestampType


# In[8]:


parsed_df = raw_df.withColumn('hour',raw_df['hour'].cast(IntegerType())).withColumn('minute',raw_df['minute'].cast(IntegerType())).withColumn('second',raw_df['second'].cast(IntegerType())).withColumn('datetime_stamp',raw_df['datetime_stamp'].cast(TimestampType())).dropna(how='any')


# ## Data Loading to BigQuery

# #### Store the parsed data from the DataFrame into the BigQuery.

# In[9]:


# Use the Google Cloud Storage bucket for temporary BigQuery export data used
# by the InputFormat. This assumes the Google Cloud Storage connector for
# Hadoop is configured.
bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')
input_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_input'.format(bucket)


# In[10]:


# Output Parameters.
output_dataset = 'nasa_dataset'
output_table = 'nasa_output'


# In[11]:


# Stage data formatted as newline-delimited JSON in Google Cloud Storage.
output_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_output'.format(bucket)
#partitions = range(word_counts.getNumPartitions())
output_files = output_directory + '/part-*'


# In[12]:


#! gsutil rm -r gs://aekanunlab/hadoop/


# In[13]:


parsed_df.write.format('csv').save(output_directory)


# In[14]:


#! bq mk nasa_dataset


# In[15]:


import subprocess
# Shell out to bq CLI to perform BigQuery import.
subprocess.check_call(
    'bq load --source_format=CSV  '
    '--replace '
    '--autodetect '
    '{dataset}.{table} {files} '
    'datetime_stamp:timestamp,day_of_month:STRING,hour:INTEGER,ip_addr:STRING,minute:INTEGER,month:STRING,second:INTEGER,timezone:INTEGER,year:INTEGER '
    .format(
        dataset=output_dataset, table=output_table, files=output_files
    ).split())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




