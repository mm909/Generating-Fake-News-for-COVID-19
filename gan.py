import pandas as pd

df = pd.read_csv("metadata.csv")

# Drop not needed cols
drops = [
        'cord_uid',
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

print(df['title'])


def generator():

    return
