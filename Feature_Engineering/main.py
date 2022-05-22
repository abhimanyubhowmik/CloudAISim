import numpy as np
import pandas as pd
from sklearn import preprocessing
from google.cloud import storage

storage_client = storage.Client(project='auto-ml-and-explainable-ai')

def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))
    
    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))
     
    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop 
    poss_drop = list(set(poss_drop).difference(set(drop)))
    
    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        
    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)
         
    return drop

def corrX(df, cut = 0.9) :
       
    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool))
    
    dropcols = list()
    
    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target', 
                                 'v2.target','corr', 'drop' ]))
    
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if(corr_mtx.iloc[row, col] > cut):
                if(avg_corr.iloc[row] > avg_corr.iloc[col]): 
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else: 
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]
                
                s = pd.Series([ corr_mtx.index[row],
                up.columns[col],
                avg_corr[row],
                avg_corr[col],
                up.iloc[row,col],
                drop],
                index = res.columns)
        
                res = res.append(s, ignore_index = True)
    
    dropcols_names = calcDrop(res)
    
    return(dropcols_names)


def upload_blob(bucket_name, source_data, destination_blob_name):
    """Uploads a file to the bucket."""    
    print('function upload_blob called')     
    bucket = storage_client.get_bucket(bucket_name)    
    blob = bucket.blob(destination_blob_name)    
    blob.upload_from_string(source_data)    
    print('File {} uploaded to {}.'.format(destination_blob_name, bucket_name))


def feature_engineering(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    filename = file['name']
    bucket = file['bucket']
    print(f"Processing file: {file['name']}.")

    csv_file = 'gs://{}/{}'.format(bucket,filename)

    df = pd.read_csv(csv_file)

    # Drop NA Values

    df.dropna(inplace= True)

    # Drop Constant Column

    for col in df.columns:
        if len(set(df[col])) == 1:
            df = df.drop(col,axis = 1) 

    # One-Hot Encoding of Object Columns

    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = preprocessing.LabelEncoder().fit_transform(df[col])

    # Drop Corellated Columns

    cor_col = corrX(df,0.8)
    df = df.drop(labels=cor_col,axis= 1)

    # Converting Table to CSV format

    data = df.to_csv(index=False)

    # Upload CSV file to Cloud Storage

    upload_blob('feature_engineering_dataset',data,filename)