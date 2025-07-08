import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/Nutrition__Physical_Activity__and_Obesity.csv")

# Filter for latest year and key questions
latest_year = df['YearStart'].max()
key_questions = [
    'Percent of adults aged 18 years and older who have obesity',
    'Percent of adults who engage in no leisure-time physical activity',
    'Percent of adults who report consuming fruit less than one time daily',
    'Percent of adults who report consuming vegetables less than one time daily',
    'Percent of adults aged 18 years and older who have an overweight classification'
]

df_filtered = df[
    (df['YearStart'] == latest_year) &
    (df['Question'].isin(key_questions)) &
    (df['Stratification1'].str.contains("18", na=False))
]

pivot_df = df_filtered.pivot_table(index='LocationDesc', columns='Question', values='Data_Value')
pivot_df.dropna(inplace=True)

# Scale & Cluster
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_df)

kmeans = KMeans(n_clusters=4, random_state=42)
pivot_df['Cluster'] = kmeans.fit_predict(scaled_data)
pivot_df.reset_index(inplace=True)


# Save results
pivot_df.to_csv("clustered_output.csv", index=False)
