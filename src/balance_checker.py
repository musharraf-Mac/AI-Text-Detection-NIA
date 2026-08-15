import pandas as pd
import os
import re
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv("../data_p/processed/TXD_22_cleaned.csv")

# Plot the distribution of text labels
df['author'].value_counts().plot(kind='bar')
plt.title('Distribution of Text Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
