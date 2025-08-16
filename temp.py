import pandas

pandas.read_parquet('pipeline_output/MindEase IBM_HACKATHON/MindEase IBM_HACKATHON.parquet').to_csv('fetch1.csv')