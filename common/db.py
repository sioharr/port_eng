import os
from sqlalchemy import create_engine as __create_engine


def create_engine_for_db(*args, **kwargs):
    username = os.environ['DB_USER']
    password = os.environ['DB_PASS']
    host = os.environ['DB_HOST']
    schema = os.environ['DB_SCHEMA']
    
    DB_URL = f"mysql://{username}:{password}@{host}/{schema}"
    return __create_engine(DB_URL, *args, **kwargs)
