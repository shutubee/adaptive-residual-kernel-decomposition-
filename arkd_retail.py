import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

SEEDS = (11, 23, 42)
MODELS = ('Seasonal-naive', 'RBF', 'GLR', 'GLRT', 'GLRT-censored')

def make_data(seed, n=300):
    rng = np.random.default_rng(seed)
    t = np.arange(n); dow = t % 7
    promo = ((t % 53) >= 42).astype(float)
    price = 3 + .25*np.sin(t/27) - .18*promo
    temp = 20 + 8*np.sin(2*np.pi*t/365)
    # Regime shift: promotions become more effective later.
    signal = (63 + 7*np.sin(2*np.pi*dow/7) + (12 + 11*(t>=195))*promo
              - 6*(price-3) + .35*(temp-20) + .8*promo*(temp-20))
    demand = np.maximum(0, signal + rng.normal(0, 4.5, n))
    stock = rng.uniform(53, 95, n)
    sales = np.minimum(demand, stock)
    censored = demand >= stock
    features = np.column_stack([np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),price,promo,temp])
    return t, features, demand, sales, stock, censored

def kernels(A, B, ta, tb, rho):
    kg = rbf_kernel(A,B,gamma=.10)
    # Gating uses raw promotion flags reconstructed as binary from standardized values.
    ga = (A[:,3] > 0).astype(float); gb = (B[:,3] > 0).astype(float)
    kl = np.outer(ga,gb)*rbf_kernel(A[:,[2,4]],B[:,[2,4]],gamma=.6)
    fa = np.column_stack([A[:,2]*A[:,3],A[:,3]*A[:,4]])/2
    fb = np.column_stack([B[:,2]*B[:,3],B[:,3]*B[:,4]])/2
    kr = fa @ fb.T
    kt = np.exp(-rho*np.abs(ta[:,None]-tb[None,:])) * kg
    return np.stack([kg,kl,kr,kt])

def train(K, y, censored, lam, censor_loss):
    n=len(y); center=np.median(y)
    mask=censored if censor_loss else np.zeros(n,dtype=bool)
    def obj(alpha):
        pred=center+K@alpha
        diff=pred-y
        active=(~mask)|(diff<0)
        err=np.where(active,diff,0)
        return (err@err/n + lam*(alpha@K@alpha),
                2*K@err/n+2*lam*K@alpha)
    result=minimize(obj,np.zeros(n),jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-8})
    return result.x,center

def score_observed(pred,sales,censored):
    # Observable validation proxy: exact loss when in stock; one-sided when stocked out.
    diff=pred-sales
    diff=np.where(censored,np.minimum(diff,0),diff)
    return np.mean(diff**2)

def run(seed):
    t,raw,demand,sales,stock,cens=make_data(seed)
    tr=np.arange(0,180); va=np.arange(180,240); te=np.arange(240,300)
    scaler=StandardScaler().fit(raw[tr]); X=scaler.transform(raw)
    out={}
    # Seasonal naive: last available same-weekday sales, measured on latent test demand.
    naive=np.array([sales[j-7] for j in te]); out['Seasonal-naive']=naive
    for name in MODELS[1:]:
        use_censor=name=='GLRT-censored'
        options=([np.array([1,0,0,0])] if name=='RBF' else
                 [np.array([.6,.2,.2,0]),np.array([.3,.5,.2,0])] if name=='GLR' else
                 [np.array([.45,.15,.15,.25]),np.array([.2,.4,.1,.3]),np.array([.2,.2,.1,.5])])
        best=None
        for rho in ([.03,.12] if 'GLRT' in name else [.03]):
            ktr=kernels(X[tr],X[tr],t[tr],t[tr],rho)
            kva=kernels(X[va],X[tr],t[va],t[tr],rho)
            kte=kernels(X[te],X[tr],t[te],t[tr],rho)
            for weights in options:
                for lam in [.01,.1]:
                    K=np.tensordot(weights,ktr,axes=1)
                    a,center=train(K,sales[tr],cens[tr],lam,use_censor)
                    pv=center+np.tensordot(weights,kva,axes=1)@a
                    val=score_observed(pv,sales[va],cens[va])
                    if best is None or val<best[0]:
                        pt=center+np.tensordot(weights,kte,axes=1)@a
                        best=(val,pt,weights,rho,lam)
        out[name]=best[1]
    return {name:(float(np.mean(np.abs(pred-demand[te]))),float(np.sqrt(np.mean((pred-demand[te])**2)))) for name,pred in out.items()},float(np.mean(cens[te]))

if __name__=='__main__':
    results=[]
    for seed in SEEDS:
        scores,stockouts=run(seed); results.append(scores)
        print(f'Seed {seed}, test stockouts {stockouts:.1%}')
        for name in MODELS:
            print(f'  {name:16s} MAE {scores[name][0]:7.3f} RMSE {scores[name][1]:7.3f}')
    print('\nAcross seeds (mean Â± sample SD of MAE):')
    for name in MODELS:
        vals=np.array([x[name][0] for x in results]); print(f'{name:16s} {vals.mean():.3f} Â± {vals.std(ddof=1):.3f}')