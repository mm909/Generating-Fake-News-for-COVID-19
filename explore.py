import pandas as pd
import numpy as np

df = pd.read_csv("metadata.csv")

# Drop not needed cols
drops = ['cord_uid',
         'sha',
         'source_x',
         'doi',
         'pmcid',
         'pubmed_id',
         'license',
         'abstract',
         'publish_time',
         'authors',
         'journal',
         'Microsoft Academic Paper ID',
         'WHO #Covidence',
         'has_pdf_parse',
         'has_pmc_xml_parse',
         'full_text_file',
         'url']

df = df.drop(drops, axis=1)
df = df.dropna()

data = df.to_numpy()
data = data.flatten()
removenums = []
for i, title in enumerate(data):
    if not title.isascii():
        removenums.append(i)
data = np.delete(data, removenums)


text = ' '
for title in data:
    text += title.lower() + "\n"
    pass

removed = ['|', '~', ':', ';', '@', '&', '_', '#', '*', '+', '`', '<', '=', '[', ']']
for char in removed:
    text = text.replace(char,'')


chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(chars)
print(char_indices)
print(indices_char)
print(len(text))
print(len(data))
