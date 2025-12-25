from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="hw5_train_from_feast",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hw5", "feast"],
) as dag:
    generate_driver_stats = BashOperator(
        task_id="generate_driver_stats",
        bash_command="python /opt/scripts/generate_driver_stats.py",
    )

    train_from_feast = BashOperator(
        task_id="train_from_feast",
        bash_command="python /opt/scripts/train_model_from_feast.py",
    )

    generate_driver_stats >> train_from_feast

