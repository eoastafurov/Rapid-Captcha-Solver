import psycopg2 as psy


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
    
    return cursor, connection


def empty_db():
    cursor, connection = load_db()
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
    connection.commit()
