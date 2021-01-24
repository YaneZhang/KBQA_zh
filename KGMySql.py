import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pymysql

username = 'username'
password = 'password'
host = 'your_db_host'
db = 'db'
connect_info = f'mysql+pymysql://{username}:{password}@{host}/{db}?charset=utf8mb4'
engine = create_engine(connect_info)

with open("qa_pipeline/data/kbqa.kb", "r", encoding='utf-8') as f:
    lines = f.readlines()
    triples = np.empty([0,3])
    for line in lines:
        triple = [i.strip() for i in line.strip().split("|||")]
        triples = np.append(triples, [triple], axis=0)

triplesDF = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
triplesDF.to_sql(name='kbqa_triples', con=engine, if_exists='replace', index=False)

