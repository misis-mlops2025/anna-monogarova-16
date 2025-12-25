from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32
from feast.value_type import ValueType

driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Driver identifier",
)

driver_stats_source = FileSource(
    path="../data/driver_stats.parquet",
    timestamp_field="event_timestamp",
)

driver_stats_fv = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=7),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Float32),
    ],
    online=True,
    source=driver_stats_source,
)

