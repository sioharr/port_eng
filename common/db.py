import os
from sqlalchemy import create_engine as __create_engine


def create_engine_for_db(*args, **kwargs):
    username = os.env['DB_USER']
    password = os.env['DB_PASS']
    host = os.env['DB_HOST']
    schema = os.env['DB_SCHEMA']
    
    DB_URL = f"mysql://{username}:{password}@{host}/{schema}"
    return __create_engine(DB_URL, *args, **kwargs)
