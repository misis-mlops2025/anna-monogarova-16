from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor

NEW_DATA_PATH = "/opt/data/new_data.csv"

def _file_exists():
    return os.path.exists(NEW_DATA_PATH)

with DAG(
    dag_id="hw4_batch_inference",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hw4"],
) as dag:
    wait_for_file = PythonSensor(
        task_id="wait_for_new_data",
        python_callable=_file_exists,
        poke_interval=5,
        timeout=60 * 60,
        mode="poke",
    )

    batch_predict = BashOperator(
        task_id="batch_predict",
        bash_command="python /opt/scripts/batch_predict.py",
    )

    wait_for_file >> batch_predict
