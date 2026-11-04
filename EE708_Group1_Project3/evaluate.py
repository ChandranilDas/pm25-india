"""Usage: python evaluate.py --dataset data/pm25_daily.csv --cells data/cells.csv"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model_io import load_model

def metrics(y, pred):
    y = np.asarray(y); error = np.asarray(pred) - y
    denom = np.sum((y-y.mean())**2)
    return {'RMSE':float(np.sqrt(np.mean(error**2))), 'MAE':float(np.mean(abs(error))), 'R2':float(1-np.sum(error**2)/denom) if denom else None, 'bias':float(np.mean(error))}

def main():
    p = argparse.ArgumentParser(description='Evaluate numeric-HDF5 random forest; PM2.5 units: micrograms/m3.')
    p.add_argument('--dataset',required=True,help='CSV with target pm25 and predictor columns')
    p.add_argument('--model',default=str(Path(__file__).with_name('Model.h5')))
    p.add_argument('--cells',help='CSV containing lat, lon, set; required unless dataset has set')
    p.add_argument('--output',default='evaluation_metrics.json')
    a = p.parse_args(); model = load_model(a.model)
    print(model)
    summary = {'trees':len(model.estimators_), 'features':list(model.feature_names_in_), 'total_nodes':sum(t.tree_.node_count for t in model.estimators_), 'total_leaves':sum(t.tree_.n_leaves for t in model.estimators_), 'max_depth':max(t.tree_.max_depth for t in model.estimators_), 'model_bytes':Path(a.model).stat().st_size}
    print('Model summary:',json.dumps(summary,indent=2,default=lambda x:x.item()))
    data = pd.read_csv(a.dataset)
    if a.cells:
        cells = pd.read_csv(a.cells)
        data = data.drop(columns='set',errors='ignore').merge(cells[['lat','lon','set']],on=['lat','lon'],how='left',validate='many_to_one')
    if 'set' not in data or data['set'].isna().any():
        raise ValueError('Supply train/test membership through --cells or a complete set column.')
    if not set(data['set']).issubset({'train','test'}): raise ValueError('set must be train or test')
    features = list(model.feature_names_in_)
    if data[features+['pm25']].isna().any().any(): raise ValueError('Missing predictors or targets')
    results = {'summary':summary,'units':'micrograms/m3','splits':{}}
    for name in ['train','test']:
        d = data[data['set']==name]
        if d.empty: continue
        start=time.perf_counter(); pred=model.predict(d[features]); elapsed=time.perf_counter()-start
        scores=metrics(d.pm25,pred)
        scores.update(rows=len(d),prediction_seconds=elapsed)
        results['splits'][name]=scores
        print(name, json.dumps(scores,indent=2))
    Path(a.output).write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
if __name__=='__main__': main()
