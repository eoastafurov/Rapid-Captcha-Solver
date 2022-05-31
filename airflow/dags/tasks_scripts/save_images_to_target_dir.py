from asyncio import shield
import cv2 
import numpy as np
import hashlib
import pickle
import psycopg2 as psy
import json
import os
from tqdm import tqdm
from typing import List

TARGET_DIR = '/home/eugeny/ibot-ml/Small'


def load_db():
    db_connect_kwargs = {
        'dbname': '****',
        'user': '****',
        'password': '****',
        'host': '****',
        'port': '****'
    }
    connection = psy.connect(**db_connect_kwargs)
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    
    return cursor, connection

def train_test_val_split(ids: List, train: float, test: float, val: float, shuffle: bool = True):
    assert train + test + val == 1
    if shuffle:
        ids = np.array(ids)
        np.random.shuffle(ids)
    total_len = len(ids)
    train_end = int(total_len * train)
    val_end = train_end + int(total_len * val)
    
    train_part = ids[:train_end]
    val_part = ids[train_end:val_end]
    test_part = ids[val_end:]
    
    return train_part, val_part, test_part
    


def save_images_to_target_dir():
    def helper(ids_array, cursor_, connection_, postfix: str):
        for id in tqdm(ids_array):
            cursor_.execute(
                """
                select id, label, bytes from images
                where id = %s
                """,
                (id,)
            )
            select_result = cursor.fetchone()
            class_name, img = select_result[1], pickle.loads(select_result[2])
            adname = os.path.join(TARGET_DIR, postfix, class_name)
            afname = os.path.join(adname, id + '.jpg')
            if not os.path.exists(adname):
                os.system('mkdir {}'.format(adname))
            cv2.imwrite(afname, img)
            assert os.path.exists(afname)
        
    cursor, connection = load_db()
    cursor.execute(
        """
        select id from images
        """ 
    )
    select_result = cursor.fetchall()
    ids = [el[0] for el in select_result]
    
    train_part, val_part, test_part = train_test_val_split(
        ids=ids, 
        train=0.8,
        val=0.2,
        test=0.0,
        shuffle=True
    )
    
    helper(train_part, cursor, connection, postfix='train')
    helper(val_part, cursor, connection, postfix='val')
    helper(test_part, cursor, connection, postfix='test')
