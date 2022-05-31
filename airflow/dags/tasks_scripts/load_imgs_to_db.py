import cv2 
import numpy as np
import hashlib
import pickle
import psycopg2 as psy
import json


def crop_aspect_ratio(img_, short_side: int = 224):
    height, width = img_.shape[0], img_.shape[1]
    rotated = False
    if height > width:
        img_ = cv2.transpose(img_)
        rotated = True
    _short_side = img_.shape[0]
    _long_side = img_.shape[1]
    ratio = _long_side / _short_side
    img_ = cv2.resize(img_, (int(short_side * ratio), short_side))
    if rotated:
        img_ = cv2.transpose(img_)
    return img_

def insert_image(img: np.ndarray, label: str, cursor_, connection_):
    im_id = hashlib.md5(img).hexdigest()
    cursor_.execute(
        """
        INSERT INTO images(id, label, bytes)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (im_id, label, pickle.dumps(img))
    )
    connection_.commit()
    
def load_db():
    db_connect_kwargs = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }
    connection = psy.connect(**db_connect_kwargs)
    connection.set_session(autocommit=True)
    cursor = connection.cursor()
    
    cursor.execute(
        """
        DROP TABLE IF EXISTS images;
        CREATE TABLE images (
            id VARCHAR(32) PRIMARY KEY,
            label VARCHAR(20),
            bytes BYTEA
        )
        """
    )
    return cursor, connection

def load_fnames():
    with open('./tmp/fnames.json', 'r') as f:
        result = json.load(f)
    return result
    

def load_imgs_to_db():
    cursor, connection = load_db()
    result = load_fnames()
    for label in result.keys():
        for afname in result[label]:
            img = cv2.imread(afname)
            if img is None:
                continue
            img = crop_aspect_ratio(img_=img, short_side=224)
            insert_image(img=img, label=label, cursor_=cursor, connection_=connection)
