from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import subprocess

def print_env_info():
    print("Python version:", sys.version)
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    print("Installed packages:\n", installed_packages.decode())

with DAG(dag_id='check_python_env', start_date=datetime(2024, 1, 1), schedule_interval=None) as dag:
    check_env_task = PythonOperator(
        task_id='check_env_task',
        python_callable=print_env_info
    )
