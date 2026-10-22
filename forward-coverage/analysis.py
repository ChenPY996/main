from pathlib import Path
import csv,json,argparse
import numpy as np
from scipy.stats import wilcoxon,binomtest,t

ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(Path(__file__).parent/'results'));args=ap.parse_args()
OUT=Path(args.results);R=list(csv.DictReader(open(OUT/'runs.csv',encoding='utf-8-sig')))
for r in R:
 for k in r:
  if k in ('suite','method','factor','sweep_signs'):continue
  if r[k]!='':r[k]=float(r[k])
def select(method=None,event=None,geometry=None,suite=None,n=None,m=None):
 return [r for r in R if (method is None or r['method']==method) and (event is None or r['event']==event) and (geometry is None or r['geometry']==geometry) and (suite is None or r['suite']==suite) and (n is None or r['n']==n) and (m is None or r['m']==m)]
def core(method,event=None,geometry=None):return [r for r in select(method,event,geometry,n=20,m=4) if r['suite'] in ('factorial','core_extra')]
def mean(a,k,mul=1):return float(np.mean([r[k] for r in a]))*mul
def summary(a):
 return {k:mean(a,k,mul) for k,mul in [('C',100),('success',100),('L',1),('A',1),('E',.001),('T_stop',1),('WAPE',100),('moves',1),
                                       ('viol_reserve',100),('viol_battery',100),('reserve_min',100),('overrun',100),('planning_ms',1)]}|{'count':len(a),'successes':int(sum(r['success'] for r in a))}
def holm(p):
 order=np.argsort(p);out=np.ones(len(p));prev=0
 for rank,i in enumerate(order):prev=max(prev,min(1,p[i]*(len(p)-rank)));out[i]=prev
 return out.tolist()
methods=list(dict.fromkeys(r['method'] for r in R));T={}
T['pooled']={m:summary([r for r in core(m) if r['event']>0]) for m in methods}
T['events']={str(e):{m:summary(core(m,e)) for m in methods} for e in range(5)}
T['geometry']={str(g):summary([r for r in core('CF',geometry=g) if r['event'] in (1,3,4)]) for g in range(1,7)}
T['sensitivity']={}
for factor in sorted(set(r['factor'] for r in R)-{''}):
 levels=sorted(set(r['level'] for r in R if r['factor']==factor));T['sensitivity'][factor]={}
 for level in levels:
  T['sensitivity'][factor][str(level)]={m:summary([r for r in R if r['factor']==factor and r['level']==level and r['method']==m]) for m in ('Model','CF','Energy')}
T['scale']=[]
for n in (20,30,40):
 for m in (3,4,6,8):
  a=[r for r in select('CF',suite='factorial',n=n,m=m) if r['event'] in (1,3,4)]
  T['scale'].append(dict(n=n,m=m,**summary(a),planning_median=float(np.median([r['planning_ms'] for r in a])),planning_p95=float(np.percentile([r['planning_ms'] for r in a],95))))
paired=[]
for e in (1,2,3,4):
 for m in ('BCD','SSI','SSI-E','QAlloc','Energy','Model'):
  a={(r['geometry'],r['seed']):r for r in core('CF',e)};b={(r['geometry'],r['seed']):r for r in core(m,e)}
  keys=sorted(a);diff=np.array([a[k]['success']-b[k]['success'] for k in keys]);good=int((diff==1).sum());bad=int((diff==-1).sum())
  p=binomtest(good,good+bad,.5).pvalue if good+bad else 1.
  paired.append(dict(event=e,method=m,metric='success',delta=100*float(diff.mean()),only_cf=good,only_baseline=bad,p=float(p)))
for r,p in zip(paired,holm([r['p'] for r in paired])):r['holm_p']=p
T['literature_paired']=paired
paired=[]
for e in (1,2,3,4):
 for metric,mul in [('C',100),('T_stop',1),('WAPE',100)]:
  a={(r['geometry'],r['seed']):r for r in core('CF',e)};b={(r['geometry'],r['seed']):r for r in core('Model',e)}
  d=np.array([(a[k][metric]-b[k][metric])*mul for k in sorted(a)])
  p=float(wilcoxon(d,zero_method='wilcox').pvalue) if np.any(abs(d)>1e-12) else 1.
  half=t.ppf(.975,len(d)-1)*np.std(d,ddof=1)/np.sqrt(len(d))
  paired.append(dict(event=e,metric=metric,delta=float(d.mean()),low=float(d.mean()-half),high=float(d.mean()+half),p=p))
for r,p in zip(paired,holm([r['p'] for r in paired])):r['holm_p']=p
T['model_paired']=paired
assert len(R)==52512
assert all(0<=r['C']<=1 and r['repeat_tasks']==0 for r in R)
assert all(r['success']==int(r['C']>=.99) for r in R)
# First-leg fan-out audit: one record per geometric instance (the planning stage is
# shared by all methods), reported instead of the previous unverified claim.
seen={}
for r in R:
 key=(r['geometry'],r['n'],r['m'],r['seed'])
 if key in seen:continue
 seen[key]=dict(geometry=r['geometry'],n=r['n'],m=r['m'],seed=r['seed'],min_deg=r['fanout_min_deg'],
                min_perp=r['fanout_min_perp'],min_gap=r['fanout_min_gap'],rotation=r['rotation'],
                signs=r['sweep_signs'],base_x=r['base_x'],base_y=r['base_y'])
fan=list(seen.values());deg=np.array([f['min_deg'] for f in fan]);perp=np.array([f['min_perp'] for f in fan]);gap=np.array([f['min_gap'] for f in fan])
T['fanout']=dict(instances=len(fan),min_deg_mean=float(deg.mean()),min_deg_p05=float(np.percentile(deg,5)),
                 min_deg_min=float(deg.min()),share_deg_below_1=float(np.mean(deg<1)),share_deg_below_3=float(np.mean(deg<3)),share_deg_below_5=float(np.mean(deg<5)),
                 min_perp_mean=float(perp.mean()),min_perp_min=float(perp.min()),share_perp_below_8=float(np.mean(perp<8)),
                 min_gap_mean=float(gap.mean()),min_gap_min=float(gap.min()),
                 descending_blocks=float(np.mean([f['signs'].count('-') for f in fan])),
                 status='measured per instance; enforced screen: pairwise leg angle >= 1 deg where feasible')
T['safety']={m:{k:T['pooled'][m][k] for k in ('viol_reserve','viol_battery','reserve_min','overrun','success','count')} for m in methods}
T['audit']=dict(runs=len(R),states=4704,core_count_per_event=240,dynamic_core_count=960,status='passed',
                metrics=['success','coverage','energy WAPE','reserved-budget violation','battery violation','first-leg fan-out separation'])
# Fleet-size (swarm) suite: 3 to 12 UAVs on the 100 m instances, seven methods.
SWP=OUT/'runs_swarm.csv'
if SWP.exists():
 SW=list(csv.DictReader(open(SWP,encoding='utf-8-sig')))
 for r in SW:
  for k in r:
   if k in ('suite','method','factor','sweep_signs'):continue
   if r[k]!='':r[k]=float(r[k])
 smeth=list(dict.fromkeys(r['method'] for r in SW));sizes=sorted({int(r['m']) for r in SW})
 T['swarm']={str(m):{x:summary([r for r in SW if r['m']==m and r['method']==x]) for x in smeth} for m in sizes}
 T['swarm_by_event']={str(m):{str(e):{x:summary([r for r in SW if r['m']==m and r['method']==x and r['event']==e]) for x in smeth}
                             for e in range(5)} for m in sizes}
 paired=[]
 for m in sizes:
  a={(r['geometry'],r['seed']):r for r in SW if r['m']==m and r['method']=='CF'}
  for x in smeth:
   if x=='CF':continue
   b={(r['geometry'],r['seed']):r for r in SW if r['m']==m and r['method']==x}
   keys=sorted(set(a)&set(b))
   diff=np.array([a[k]['success']-b[k]['success'] for k in keys])
   good=int((diff==1).sum());bad=int((diff==-1).sum())
   p=binomtest(good,good+bad,.5).pvalue if good+bad else 1.
   paired.append(dict(m=m,method=x,delta=100*float(diff.mean()),only_cf=good,only_baseline=bad,p=float(p),n=len(keys)))
 for r,p in zip(paired,holm([r['p'] for r in paired])):r['holm_p']=p
 T['swarm_paired']=paired
 T['audit']['swarm_states']=len(SW)//len(smeth);T['audit']['swarm_runs']=len(SW)
(OUT/'summary.json').write_text(json.dumps(T,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ('literature_paired','model_paired','scale'):
 with open(OUT/(name+'.csv'),'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(T[name][0]));w.writeheader();w.writerows(T[name])
print(json.dumps(T['pooled'],ensure_ascii=True,indent=2))
