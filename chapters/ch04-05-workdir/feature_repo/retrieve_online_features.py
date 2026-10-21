from feast import FeatureStore

FEATURE_LIST = [
    "demographic:Sex",
    "demographic:Race",
    "demographic:Native_country",
    "relationship:Relationship",
    "relationship:Marital-Status",
    "occupation:Workclass",
    "occupation:Education",
    "occupation:Occupation",
]
store = FeatureStore(repo_path=".")
online_features = store.get_online_features(
    features=FEATURE_LIST,
    entity_rows=[{"user_id": "9f2ac416-06e1-44a0-87bd-d4787c85bf66"}],  # B
)

print(online_features.to_df())
