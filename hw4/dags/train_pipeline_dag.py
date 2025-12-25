from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="hw4_train_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hw4"],
) as dag:
    generate_data = BashOperator(
        task_id="generate_data",
        bash_command="python /opt/scripts/generate_data.py",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="python /opt/scripts/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        bash_command="python /opt/scripts/train.py",
    )

    generate_data >> preprocess >> train
