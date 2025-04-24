import pandas as pd

# Load your Excel file
df = pd.read_excel("/workspaces/Model_development/Data/set2.xlsx")

# Convert all columns except the last three (assuming y1, y2, y3) to string
feature_columns = df.columns[:-3]
df[feature_columns] = df[feature_columns].astype(str)

# Save to a new Excel file
df.to_excel("set2_all_features_as_string.xlsx", index=False)
