import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def simulate(seed=11, n=320):
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    promo=(rng.random(n)<.22).astype(float)
    price=3.0+0.3*np.sin(t/17)-0.45*promo+rng.normal(0,.08,n)
    temp=19+9*np.sin(2*np.pi*t/90)+rng.normal(0,2,n)
    weekend=(t%7>=5).astype(float)
    X=np.column_stack([price,temp,promo,weekend,np.sin(2*np.pi*t/7),np.cos(2*np.pi*t/7)])
    shift=(t>=190).astype(float)
    demand=np.maximum(0,40+7*weekend+15*promo+0.35*(temp-19)-5*(price-3)+shift*(10+12*promo)+rng.normal(0,4,n))
    inventory=np.maximum(8, demand+rng.normal(-2,9,n))
    sales=np.minimum(demand,inventory)
    censored=demand>=inventory
    return X,t,sales,censored,demand


class CensoredKernel:
    def __init__(self, adaptive=True, lam=0.8, sigma=6.0, n_regimes=3, seed=0):
        self.adaptive=adaptive; self.lam=lam; self.sigma=sigma
        self.n_regimes=n_regimes; self.seed=seed

    def _features(self,X,t,fit=False):
        if fit:
            self.scaler=StandardScaler().fit(X)
        Z=self.scaler.transform(X)
        if fit:
            self.kmeans=KMeans(n_clusters=self.n_regimes,random_state=self.seed,n_init=10).fit(Z)
        dist=self.kmeans.transform(Z)
        gates=np.exp(-dist**2/4)
        gates/=gates.sum(axis=1,keepdims=True)
        return Z,gates

    def _kernel(self,Z,A,t,Z2,B,t2):
        G=rbf_kernel(Z,Z2,gamma=.09)
        L=rbf_kernel(Z,Z2,gamma=.8)
        T=np.exp(-np.abs(t[:,None]-t2[None,:])/25)*G
        if self.adaptive:
            R=sum((A[:,r,None]*B[None,:,r])*rbf_kernel(Z,Z2,gamma=.35+0.25*r)
                  for r in range(self.n_regimes))
        else:
            R=rbf_kernel(Z,Z2,gamma=.35)
        return .25*G+.25*L+.15*T+.35*R

    def fit(self,X,t,y,censored):
        Z,A=self._features(X,t,fit=True)
        self.Z=Z;self.A=A;self.t=np.asarray(t)
        self.ymean=float(np.mean(y));self.yscale=max(float(np.std(y)),1.)
        target=(y-self.ymean)/self.yscale
        K=self._kernel(Z,A,self.t,Z,A,self.t)
        # Cholesky-inspired ridge penalty, stable for PSD K
        K=(K+K.T)/2
        def objective(alpha):
            mu=K@alpha
            z=(target-mu)/(self.sigma/self.yscale)
            unc=~censored
            # censored observation: P(D >= observed sales)
            loss=.5*np.sum(z[unc]**2)-np.sum(log_ndtr(-z[censored]))
            loss+=.5*self.lam*alpha@(K@alpha)
            return float(loss)
        result=minimize(objective,np.zeros(len(y)),method='L-BFGS-B',options={'maxiter':130})
        self.alpha=result.x
        self.optimization_success=bool(result.success)
        return self

    def predict(self,X,t):
        Z,A=self._features(X,t)
        K=self._kernel(Z,A,np.asarray(t),self.Z,self.A,self.t)
        return np.maximum(0,self.ymean+self.yscale*(K@self.alpha))


def run(seed):
    X,t,sales,censored,true=simulate(seed)
    tr=np.arange(0,210); te=np.arange(210,len(t))
    scaler=StandardScaler().fit(X[tr])
    Xt=scaler.transform(X[tr]); Xv=scaler.transform(X[te])
    models={
      'Fixed-kernel censored':CensoredKernel(adaptive=False,seed=seed),
      'Regime-gated censored':CensoredKernel(adaptive=True,seed=seed),
      'Random forest (sales)':RandomForestRegressor(n_estimators=160,min_samples_leaf=4,random_state=seed),
      'Gradient boosting (sales)':HistGradientBoostingRegressor(max_iter=130,l2_regularization=2,random_state=seed)
    }
    results=[]
    for name,m in models.items():
        if isinstance(m,CensoredKernel):
            m.fit(X[tr],t[tr],sales[tr],censored[tr]); pred=m.predict(X[te],t[te])
        else:
            m.fit(Xt,sales[tr]);pred=np.maximum(0,m.predict(Xv))
        mae=mean_absolute_error(true[te],pred)
        # Point forecast order-up-to: intentionally same policy for every model
        q=pred+5
        cost=np.mean(2*np.maximum(true[te]-q,0)+np.maximum(q-true[te],0))
        stockout=np.mean(true[te]>q)
        results.append((seed,name,mae,cost,stockout))
    return results


if __name__=='__main__':
    import csv
    all_rows=[]
    for seed in [11,23,42]:
        all_rows.extend(run(seed))
    with open('ca_arkd_v7_results.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','model','latent_demand_mae','inventory_cost','stockout_rate']);w.writerows(all_rows)
    for name in sorted(set(r[1] for r in all_rows)):
        a=np.array([[r[2],r[3],r[4]] for r in all_rows if r[1]==name])
        print(f'{name:25s} MAE={a[:,0].mean():.3f} Cost={a[:,1].mean():.3f} Stockout={a[:,2].mean():.1%}')
    print('Saved ca_arkd_v7_results.csv')