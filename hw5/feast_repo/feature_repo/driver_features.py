from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Entity — водитель
driver = Entity(name="driver", join_keys=["driver_id"])

# Источник — паркетный файл с фичами
driver_stats_source = FileSource(
    path="/opt/data/driver_stats.parquet",
    timestamp_field="event_timestamp",
)

# FeatureView: conv_rate, acc_rate, avg_daily_trips
driver_stats_fv = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=365),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Float32),
    ],
    online=True,
    source=driver_stats_source,
)

