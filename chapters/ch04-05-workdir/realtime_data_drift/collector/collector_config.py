from evidently.collector.config import (
    CollectorConfig,
    IntervalTrigger,
    ReportConfig,
    CollectorServiceConfig,
)


config = CollectorServiceConfig(
    collectors={
        "income-classifier": CollectorConfig(
            trigger=IntervalTrigger(interval=60 * 60),
            report_config=ReportConfig.from_report(report),
            reference_path="reference_data.parquet",
            project_id="834ec9a0-ee58-4e64-816b-c593b0b6c45c",
            api_url="http://localhost:8000",
        )
    }
)


config.save("collector_config.json")
