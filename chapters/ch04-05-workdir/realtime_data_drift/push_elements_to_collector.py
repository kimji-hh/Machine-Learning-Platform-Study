from evidently.collector.client import CollectorClient
from sklearn import datasets

client = CollectorClient("http://localhost:8001")
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_cur = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

current_data = adult_cur.iloc[100 * 2 : 100 * (2 + 1), :]

client.send_data("income-classifier", current_data)
