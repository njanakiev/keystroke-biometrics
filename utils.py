import pandas as pd
from sklearn.decomposition import PCA

def load_data():
    df = pd.read_csv('data/DSL-StrongPasswordData.csv')

    H_columns  = [col for col in df.columns if col.startswith('H')]
    DD_columns = [col for col in df.columns if col.startswith('DD')]
    UD_columns = [col for col in df.columns if col.startswith('UD')]

    data = {}
    data['total'] = df.drop(columns=['subject', 'sessionIndex', 'rep'])
    data['H']     = df[H_columns]
    data['DD']    = df[DD_columns]
    data['UD']    = df[UD_columns]
    data['pca3']  = pd.DataFrame(PCA(n_components=3).fit_transform(data['total']))
    data['pca10'] = pd.DataFrame(PCA(n_components=10).fit_transform(data['total']))
    
    return data, df['subject'].values
