import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

SEEDS=(11,23,42)

def generate(seed,n=320):
    rng=np.random.default_rng(seed); t=np.arange(n)
    price=3+.25*np.sin(t/19)+rng.normal(0,.1,n)
    promo=(rng.random(n)<.16).astype(float)
    weather=20+7*np.sin(t/33)+rng.normal(0,2,n)
    dow=t%7
    demand=np.maximum(1,34+7*np.sin(2*np.pi*dow/7)+.22*weather-4*(price-3)+10*promo+4*np.sin(t/17)+rng.normal(0,4,n))
    capacity=np.where(rng.random(n)<.34, demand*rng.uniform(.55,.95,n),demand+rng.uniform(5,18,n))
    sales=np.minimum(demand,capacity)
    censored=capacity<demand
    X=np.column_stack([price,promo,weather,np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),t/n])
    return X,demand,sales,censored,capacity,t

def components(X,Z):
    # Standardized feature vectors; global and local bandwidths, interaction residual,
    # temporal recency. Every component is positive semidefinite.
    global_k=rbf_kernel(X[:,:5],Z[:,:5],gamma=.12)
    local_k=rbf_kernel(X[:,:5],Z[:,:5],gamma=1.2)
    phi=lambda a: np.column_stack([a[:,0]*a[:,1],a[:,1]*a[:,2],a[:,0]*a[:,2]])
    residual=phi(X)@phi(Z).T/3
    temporal=np.exp(-3*np.abs(X[:,5,None]-Z[None,:,5]))*global_k
    return [global_k,local_k,residual,temporal]

def kernel(X,Z,w):
    return sum(float(a)*b for a,b in zip(w,components(X,Z)))

def fit_censored(X,y,censored,w=(.35,.35,.2,.1),lam=.4):
    K=kernel(X,X,w); n=len(y); baseline=float(np.mean(y[~censored])); target=y-baseline; scale=max(np.std(y),1.)
    # RKHS representer objective with Gaussian uncensored likelihood and
    # survival likelihood for censored observations.
    def loss(z):
        a=z[:-1]; sigma=np.exp(z[-1]); mu=baseline+K@a
        u=~censored
        ll=np.sum(norm.logpdf(y[u],mu[u],sigma))
        ll+=np.sum(norm.logsf((y[censored]-mu[censored])/sigma))
        return -ll+lam*(a@K@a)/2
    z0=np.r_[np.linalg.solve(K+2*np.eye(n),target),np.log(scale)]
    res=minimize(loss,z0,method='L-BFGS-B',bounds=[(None,None)]*n+[(np.log(.5),np.log(30))],options={'maxiter':110})
    return res.x[:-1],np.exp(res.x[-1]),res.success,baseline

def inventory_cost(q,d,under=3.,over=1.):
    return np.mean(under*np.maximum(d-q,0)+over*np.maximum(q-d,0))

def experiment():
    rows=[]; curve=None
    for seed in SEEDS:
        X,d,y,c,cap,t=generate(seed)
        split=210; valid=265
        scaler=StandardScaler().fit(X[:split,:5]); Xs=X.copy(); Xs[:,:5]=scaler.transform(X[:,:5]);
        # temporal coordinate is preserved in [0,1]
        weights=[(.25,.45,.2,.1),(.5,.25,.15,.1),(.2,.3,.3,.2)]
        best=None
        for w in weights:
            a,s,_,b=fit_censored(Xs[:split],y[:split],c[:split],w)
            pred=b+kernel(Xs[split:valid],Xs[:split],w)@a
            # Censoring-aware validation objective, not inaccessible latent demand
            val=np.mean(np.where(c[split:valid],np.maximum(y[split:valid]-pred,0)**2,(y[split:valid]-pred)**2))
            if best is None or val<best[0]:best=(val,w)
        w=best[1]
        a,s,ok,b=fit_censored(Xs[:valid],y[:valid],c[:valid],w)
        pred=b+kernel(Xs[valid:],Xs[:valid],w)@a
        # For a Gaussian demand forecast, choose critical fractile; clipped at zero
        q=np.maximum(0,pred+norm.ppf(.75)*s)
        rf=RandomForestRegressor(n_estimators=120,min_samples_leaf=4,random_state=seed)
        rf.fit(Xs[:valid],y[:valid]); rfp=rf.predict(Xs[valid:]);
        # baseline trained on observed sales; no privileged true demand
        for name,p,order in [('ARKD-censored',pred,q),('Random Forest (sales)',rfp,np.maximum(0,rfp+norm.ppf(.75)*max(np.std(y[:valid]-rf.predict(Xs[:valid])),1)))]:
            rows.append(dict(seed=seed,model=name,latent_MAE=np.mean(abs(d[valid:]-p)),inventory_cost=inventory_cost(order,d[valid:]),stockout_rate=np.mean(d[valid:]>order),mean_order=np.mean(order),weight_global=w[0] if name.startswith('ARKD') else np.nan,weight_local=w[1] if name.startswith('ARKD') else np.nan,weight_residual=w[2] if name.startswith('ARKD') else np.nan,weight_temporal=w[3] if name.startswith('ARKD') else np.nan,fit_converged=ok if name.startswith('ARKD') else np.nan))
        if seed==11:curve=(t[valid:],d[valid:],y[valid:],pred,q)
    out=pd.DataFrame(rows);out.to_csv('/mnt/data/arkd_v6_results.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4)); tt,dd,yy,pp,qq=curve
    ax.plot(tt,dd,label='Latent demand (simulation only)');ax.plot(tt,yy,label='Observed sales');ax.plot(tt,pp,label='ARKD forecast');ax.plot(tt,qq,label='Order-up-to level');ax.set_xlabel('Day');ax.set_ylabel('Units');ax.legend(ncol=2);fig.tight_layout();fig.savefig('/mnt/data/arkd_v6_forecast.png',dpi=160);plt.close(fig)
    print(out.groupby('model')[['latent_MAE','inventory_cost','stockout_rate']].mean().round(3).to_string())
    print('Weights selected:',out[out.model=='ARKD-censored'][['seed','weight_global','weight_local','weight_residual','weight_temporal']].to_string(index=False))
if __name__=='__main__':experiment()