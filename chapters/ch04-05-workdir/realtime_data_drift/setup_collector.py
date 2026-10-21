import datetime

from sklearn import datasets

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_cur = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]


from evidently.collector.config import (
    CollectorConfig,
    IntervalTrigger,
    ReportConfig,
    CollectorServiceConfig,
)

data_drift_report = Report(
    metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name="age", stattest="wasserstein"),
        ColumnSummaryMetric(column_name="age"),
        ColumnDriftMetric(column_name="education-num", stattest="wasserstein"),
        ColumnSummaryMetric(column_name="education-num"),
    ],
    timestamp=datetime.datetime.now() + datetime.timedelta(days=0),
)

data_drift_report.run(
    reference_data=adult_ref,
    current_data=adult_cur.iloc[100 * 0 : 100 * (0 + 1), :],
)

config = CollectorServiceConfig(
    collectors={
        "income-classifier": CollectorConfig(
            trigger=IntervalTrigger(interval=60),
            report_config=ReportConfig.from_report(data_drift_report),
            project_id="e328fbc0-c43f-4e2d-8e45-ec49cc131caf",
            api_url="http://localhost:8005",
        )
    }
)


config.save("collector_config.json")
