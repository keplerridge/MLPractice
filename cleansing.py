#%%
import pandas as pd
from dotenv import load_dotenv
import os

#%%
load_dotenv()
data_url = os.getenv('DATA_URL')
df = pd.read_csv(data_url, compression = 'gzip', sep = '\t')

# %%
# Filter out all samples that don't have 1-22, X, or Y as a value
valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
df = df[df['Chromosome'].isin(valid_chromosomes)]

# %%
# Filter out all values tat don't have protein_coding in Gene_Biotype
df = df[df['Gene_Biotype'] == 'protein_coding']

# %%
cols_to_drop = ['Dataset_ID', 'HGNC_Symbol', 'Ensembl_Gene_ID', 'Chromosome', 'Gene_Biotype']
df = df.drop(columns = cols_to_drop)

# %%
# Find duplicate values of Entrez_Gene_ID and average them together
df = df.groupby('Entrez_Gene_ID').mean().reset_index()

# %%
# Transpose dataframe
df.set_index('Entrez_Gene_ID', inplace=True)
df_transposed = df.transpose()

# %%
meta_df = pd.read_csv('https://osf.io/download/zjry7', sep = '\t')

# %%
# Rename er_status_ihc to Class
meta_df.rename(columns = {'er_status_ihc': 'Class'}, inplace = True)

# %%
# Drop all columns except for the newly made class column
meta_df = meta_df.loc[:, ['Class', 'Sample_ID']]

# %%
# Reset the index so that it isn't just the GSM identifiers
df_transposed = df_transposed.reset_index()
df_transposed.rename(columns={'index': 'Sample_ID'}, inplace=True)

# %%
# Merge the metadata dataframe with the dataset
merged_df = pd.merge(meta_df, df_transposed, on='Sample_ID', how='inner')

# %%
# Drop the Sample_ID column
merged_df = merged_df.drop(columns = ['Sample_ID'])

# %%
# Convert all class values to 1 or 0 and drop the samples without 1 or 0
merged_df['Class'] = merged_df['Class'].replace({'P': 1, 'N': 0})
merged_df = merged_df[merged_df['Class'].isin([0, 1])]

# %%
merged_df.to_csv('./ml_ready.tsv', sep = '\t')
# %%
