from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="hw5_batch_inference_from_feast",
    start_date=datetime(2025, 1, 1),
    schedule=None,          # только ручной запуск
    catchup=False,
    tags=["hw5", "feast", "batch"],
) as dag:
    batch_inference = BashOperator(
        task_id="batch_inference_from_feast",
        bash_command="""
python /opt/scripts/batch_inference_from_feast.py {% if 
dag_run.conf.get('as_of') %} --as-of {{ dag_run.conf['as_of'] }} {% endif %}
""",
    )

