from evidently.ui.remote import RemoteWorkspace
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
from evidently.ui.remote import RemoteWorkspace, RemoteProject
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

ws = RemoteWorkspace("http://localhost:8000")
project = ws.create_project(
    name="Income Classifier",
    description="Used to classify users into multiple income bands",
)

title_panel = DashboardPanelCounter(
    filter=ReportFilter(metadata_values={}, tag_values=[]),
    agg=CounterAgg.NONE,
    title="Income Classifier Batch Data Drift Dashboard",
)

percentage_of_drifted_columns_panel = DashboardPanelCounter(
    title="Share of Drifted Features",
    filter=ReportFilter(metadata_values={}, tag_values=[]),
    value=PanelValue(
        metric_id="DatasetDriftMetric",
        field_path="share_of_drifted_columns",
        legend="share",
    ),
    text="share",
    agg=CounterAgg.LAST,
    size=1,
)

project.dashboard.add_panel(title_panel)
project.dashboard.add_panel(percentage_of_drifted_columns_panel)
project.save()
