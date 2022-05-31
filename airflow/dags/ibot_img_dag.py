from airflow import DAG
from datetime import datetime

from airflow.operators.python_operator import PythonOperator

from tasks_scripts.load_fnames import load_fnames
from tasks_scripts.load_imgs_to_db import load_imgs_to_db
from tasks_scripts.empty_target_dir import empty_target_dir
from tasks_scripts.save_images_to_target_dir import save_images_to_target_dir
from tasks_scripts.empty_db import empty_db

default_args = {
    'start_date': datetime(2020, 1, 1)
}


with DAG(
    'ibot_img_dag',
    schedule_interval='@daily', 
    default_args=default_args,
    catchup=False
) as dag:
    load_filenames_to_json_task = PythonOperator(
        task_id='load_filenames_to_json',
        python_callable=load_fnames
    )
    
    load_imgs_to_db_task = PythonOperator(
        task_id='load_imgs_to_db',
        python_callable=load_imgs_to_db
    )
    
    empty_target_dir_task = PythonOperator(
        task_id='empty_target_dir',
        python_callable=empty_target_dir
    )
    
    save_images_to_target_dir_task = PythonOperator(
        task_id='save_images_to_target_dir',
        python_callable=save_images_to_target_dir
    )
    
    empty_db_task = PythonOperator(
        task_id='empty_db',
        python_callable=empty_db
    )
    
    load_filenames_to_json_task >> load_imgs_to_db_task >> empty_target_dir_task
    empty_target_dir_task >> save_images_to_target_dir_task >> empty_db_task
