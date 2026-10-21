import os

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

if os.getenv("ENV_NAME") == "production":
    ENV_FILE_NAME = "production.env"
else:
    ENV_FILE_NAME = ".env"
