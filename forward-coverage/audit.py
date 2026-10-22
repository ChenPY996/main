"""Re-execute the published experiment and audit routes and numeric results.

Timings depend on the machine and are intentionally excluded from equality.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,csv,json,math,time
import numpy as np
from matplotlib.path import Path as Polygon
from simulation import configs,simulate,path_stats,demand,angle,DEFAULT

EXPECTED={}
def key(r):
    return (r['suite'],int(r['geometry']),int(r['event']),int(r['n']),int(r['m']),int(r['seed']),r['method'],r.get('factor',''),str(r.get('level','')))
def init(csvpath):
    global EXPECTED
    EXPECTED={key(r):r for r in csv.DictReader(open(csvpath,encoding='utf-8-sig'))}
def close(x,y,strict=False):
    # Acos of an almost-collinear float vector can add tiny spurious angles
    # when line-by-line metrics are compared with the DP's aggregate span.
    assert math.isclose(float(x),float(y),rel_tol=2e-9 if strict else 1e-7,abs_tol=2e-7 if strict else 2e-3),(x,y)
def path_check(points,rotation,sign=1):
    a=np.asarray(points); uv=a@rotation.T
    # The launch leg runs from the external base to the entry row of that block and
    # is not part of the sweep frontier; the frontier is monotone from entry on.
    assert np.all(sign*np.diff(uv[1:,1])>=-1e-7),'backward sweep frontier'
    edges=np.diff(a,axis=0)
    assert all(angle(x,y)<1.999999 for x,y in zip(edges,edges[1:])), 'immediate reversal'
    keys=[tuple(sorted((tuple(np.round(x,7)),tuple(np.round(y,7))))) for x,y in zip(a,a[1:])]
    assert len(keys)==len(set(keys)), 'repeated executed edge'
def worker(cfg):
    rows,dd=simulate(cfg,True); par=DEFAULT|cfg.get('par',{});m=cfg['m']
    for row in rows:
        ref=EXPECTED[key(row)]
        for k,v in row.items():
            if k in ('planning_ms','preprocessing_ms','suite','method','factor','level','sweep_signs'):continue
            close(v,ref[k],True)
    z=next(iter(dd.values()));reg=z['region'];base=np.asarray(reg.base)
    assert not Polygon(reg.polygon).contains_point(base)
    initial=z['initial']; directions=[np.array(r['points'][1])-base for r in initial]
    assert len({r['ids'][0] for r in initial})==m
    for i,u in enumerate(directions):
        for v in directions[i+1:]:
            cross=abs(float(u[0]*v[1]-u[1]*v[0]))
            assert cross>1e-8*np.linalg.norm(u)*np.linalg.norm(v) or np.dot(u,v)<=0,'overlapping approach'
    for method,z in dd.items():
        allids=[k for job in z['alloc'] for k in job]+[k for ids in z['prefix_ids'] for k in ids]
        assert len(allids)==len(set(allids))==len(reg.all) and set(allids)==reg.all
        visits=[k for ids in z['visited'] for k in ids]
        assert len(visits)==len(set(visits))
        total_e=0.; total_l=0.; total_a=0.;times=[];pred=[];truth=[]
        for i,(path,plan,ids) in enumerate(zip(z['paths'],z['plans'],z['visited'])):
            assert tuple(path[0])==reg.base
            assert len(path)==len(ids)+1
            assert plan['ids'] and set(plan['ids'])==z['alloc'][i] or not plan['ids'] and not z['alloc'][i]
            path_check(path,reg.rotation,z['signs'][i])
            path_check(z['prefix'][i]+plan['points'][1:],reg.rotation,z['signs'][i])
            ll,aa,_=path_stats(plan['points'],z['headings'][i]);close(ll,plan['L']);close(aa,plan['A'])
            pll,paa,_=path_stats(z['prefix'][i]);pre,pt=demand(pll,paa,z['q0'][i],z['theta'][i])
            idle=max(0,z['event_t']-pt);pre+=z['theta'][i]*20*idle;close(pre,z['spent'][i])
            capacity=1.6*z['battery_basis'][i]
            assert pre<=capacity+1e-7,'prefix including synchronization exceeds initial budget'
            expected_b=capacity-pre
            if cfg['event'] in (1,3,4) and i==m-1:expected_b*=par['battery_factor']
            close(expected_b,z['battery'][i])
            suffix=path[len(z['prefix'][i])-1:]
            sl,sa,_=path_stats(suffix,z['headings'][i]);ee,tt=demand(sl,sa,z['q'][i],z['true_theta'][i])
            assert ee<=z['battery'][i]+1e-7
            total_e+=pre+ee;times.append(z['event_t']+tt)
            l,a,_=path_stats(path);total_l+=l;total_a+=a
            truth.append(demand(ll,aa,z['q'][i],z['true_theta'][i])[0])
            factor=z['observed'][i] if method in ('CF','NoRisk','NoComm','NoTurn','NoSplit','NoSafety') else 1.
            pred.append(demand(ll,aa,z['q'][i],factor,method=='NoTurn',method=='NoComm')[0])
        for k,v in [('C',len(visits)/len(reg.all)),('L',total_l),('A',total_a),('E',total_e),('T_stop',max(times)),('WAPE',np.abs(np.array(pred)-truth).sum()/max(sum(truth),1))]:close(v,z['result'][k])
        for tr in z['trace']:
            if tr['phase']=='overload':assert tr['phi_after']<tr['phi_before']-1e-9
            else:assert tr['phi_after']<=1e-9 and tr['J_after']<tr['J_before']-1e-9
    return len(rows)

if __name__=='__main__':
    here=Path(__file__).parent
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(here/'final_results' if (here/'final_results').exists() else here/'results'));ap.add_argument('--workers',type=int,default=6);ap.add_argument('--sample',action='store_true');ap.add_argument('--sequential',action='store_true');a=ap.parse_args()
    dest=Path(a.results);cc=configs(False)
    if a.sample:cc=[c for c in cc if c['suite']=='factorial' and c['seed']==314159]
    start=time.perf_counter();runs=0
    if a.sequential:
        init(str(dest/'runs.csv'))
        for i,c in enumerate(cc,1):
            runs+=worker(c)
            if i%200==0:print(f'audited {i}/{len(cc)} states, {runs} runs',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(dest/'runs.csv'),)) as pool:
            for i,n in enumerate(pool.map(worker,cc),1):
                runs+=n
                if i%200==0:print(f'audited {i}/{len(cc)} states, {runs} runs',flush=True)
    result=dict(status='passed',states=len(cc),runs=runs,elapsed_s=time.perf_counter()-start,all_states=not a.sample,
        checks=['deterministic numeric reproduction except timing','external common base','distinct nonoverlapping nominal straight approaches','quantified first-leg fan-out separation (min angle / min perpendicular offset)','task ownership conservation','unique task visits','monotone per-UAV sweep frontier (increasing or decreasing v)','no immediate 180-degree reversal','no repeated executed edge','planned route metrics','prefix/hover/event/post-event energy accounting','actual budget and coverage','reserved-budget and battery violation audit','strict potential or objective descent'],
        limitations=['No inter-UAV collision scheduling or continuous sensor-footprint coverage guarantee.','Approach nonoverlap is audited for nominal launch segments; dynamic trajectories may cross in space.'])
    (dest/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
