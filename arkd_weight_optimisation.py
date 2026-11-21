import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler
from arkd_retail_v2 import make_data,kernels

SEEDS=(11,23,42)
WEIGHTS=np.vstack([np.eye(4),np.array([[.25,.25,.25,.25],[.45,.28,.17,.10],[.6,.2,.2,0]]) ,np.random.default_rng(2026).dirichlet([1,1,1,1],size=48)])
FOLDS=[(np.arange(120),np.arange(120,150)),(np.arange(150),np.arange(150,180)),(np.arange(180),np.arange(180,210)),(np.arange(210),np.arange(210,240))]

def fit_predict(Ktr,Kval,y,lam):
    center=np.median(y)
    cf=cho_factor(Ktr+lam*np.eye(len(y)),lower=True,check_finite=False)
    return center+Kval@cho_solve(cf,y-center,check_finite=False)

def run(seed):
    t,raw,demand,sales,stock,cens=make_data(seed)
    X=StandardScaler().fit(raw[:120]).transform(raw)
    records=[]
    for rho in (.015,.05,.12):
        matrices=[]
        for tr,va in FOLDS:
            matrices.append((kernels(X[tr],X[tr],t[tr],t[tr],rho),kernels(X[va],X[tr],t[va],t[tr],rho),tr,va))
        for lam in (0.1,1.0):
            for w in WEIGHTS:
                errors=[]
                for ktr,kva,tr,va in matrices:
                    pred=fit_predict(np.tensordot(w,ktr,axes=1),np.tensordot(w,kva,axes=1),sales[tr],lam)
                    # Censored observations: sales are a lower bound on demand.
                    e=np.where(cens[va],np.minimum(pred-sales[va],0),pred-sales[va])
                    errors.append(np.mean(e**2))
                records.append((np.mean(errors),rho,lam,w.copy()))
    records.sort(key=lambda a:a[0]); val,rho,lam,w=records[0]
    tr=np.arange(240); te=np.arange(240,300)
    kt=kernels(X[tr],X[tr],t[tr],t[tr],rho); kv=kernels(X[te],X[tr],t[te],t[tr],rho)
    pred=fit_predict(np.tensordot(w,kt,axes=1),np.tensordot(w,kv,axes=1),sales[tr],lam)
    return dict(seed=seed,weights=w.tolist(),rho=rho,lam=lam,validation_proxy_mse=float(val),test_latent_mae=float(np.mean(abs(pred-demand[te]))),test_stockout_fraction=float(np.mean(cens[te])))

if __name__=='__main__':
    import json
    results=[run(seed) for seed in SEEDS]
    print(json.dumps(results,indent=2))
    with open('/mnt/data/arkd_optimized_weights.json','w') as f:json.dump(results,f,indent=2)
