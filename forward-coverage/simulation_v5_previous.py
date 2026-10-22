"""Obstacle-free polygon coverage with fan-out launch and forward sweep routes.

Only task observations count as coverage. Transit may cross the boundary with
the camera disabled: the target boundary is not a flight prohibition.
"""
from __future__ import annotations
import argparse,csv,json,time,math,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import norm
from matplotlib.path import Path as Polygon

BASE_SEED=314159
METHODS=('Fixed','BCD','SSI','SSI-E','QAlloc','Energy','Model','CF','NoRisk','NoComm','NoTurn','NoSplit','NoSafety')
DEFAULT=dict(sigma=.20,noise=.035,battery_factor=.35,comm=.20,drift=.20,margin=.05,delta=.05,util_weight=.15,event_fraction=.30,block_size=6)
SHAPES={
1:('Skew pentagon',[(.10,.18),(.73,.07),(.95,.62),(.62,.93),(.08,.76)]),
2:('Slender quadrilateral',[(.06,.25),(.88,.06),(.96,.42),(.14,.64)]),
3:('L-shaped',[(.08,.08),(.91,.08),(.91,.34),(.37,.34),(.37,.91),(.08,.91)]),
4:('U-shaped',[(.08,.08),(.93,.08),(.93,.93),(.69,.93),(.69,.37),(.33,.37),(.33,.93),(.08,.93)]),
5:('C-shaped',[(.08,.08),(.93,.08),(.93,.31),(.36,.31),(.36,.68),(.93,.68),(.93,.92),(.08,.92)]),
6:('Five-arm star',[(.5+.43*math.cos(math.pi/2+2*math.pi*k/10) if k%2==0 else .5+.21*math.cos(math.pi/2+2*math.pi*k/10),
                       .5+.43*math.sin(math.pi/2+2*math.pi*k/10) if k%2==0 else .5+.21*math.sin(math.pi/2+2*math.pi*k/10)) for k in range(10)])}

def angle(a,b):
    if a is None or b is None:return 0.
    an=math.hypot(*a);bn=math.hypot(*b)
    if an<1e-10 or bn<1e-10:return 0.
    return math.acos(max(-1.,min(1.,(a[0]*b[0]+a[1]*b[1])/(an*bn))))/(math.pi/2)

def path_stats(points,previous=None):
    length=0.;turns=0.;heading=previous
    for a,b in zip(points,points[1:]):
        d=(b[0]-a[0],b[1]-a[1]);length+=math.hypot(*d);turns+=angle(heading,d);heading=d
    return length,turns,heading

def demand(length,turns,q,theta=1.,ignore_turn=False,ignore_comm=False):
    turns=0. if ignore_turn else turns
    t0=length/5+turns
    wait=0. if ignore_comm else .1*(1/q-1)*t0
    return theta*(10*length+50*turns+20*wait),t0+wait

class Region:
    def __init__(self,geometry,n,seed,rotation=None):
        self.geometry=geometry;self.n=n;self.h=5.;side=n*5
        rng=np.random.default_rng(seed+geometry*19073)
        vv=np.array(SHAPES[geometry][1])*side
        # Small affine variability preserves topology and the six shape families.
        jitter=rng.uniform(-12,12)*math.pi/180
        rot=np.array([[math.cos(jitter),-math.sin(jitter)],[math.sin(jitter),math.cos(jitter)]])
        vv=(vv-side*.5)@rot.T*np.array([rng.uniform(.97,1.03),rng.uniform(.97,1.03)])+side*.5
        self.polygon=vv;self.base=(-.18*side,-.18*side)
        candidates=[]
        angles=[rotation] if rotation is not None else list(range(0,180,15))
        for degrees in angles:
            rad=degrees*math.pi/180
            matrix=np.array([[math.cos(rad),math.sin(rad)],[-math.sin(rad),math.cos(rad)]])
            uv=vv@matrix.T; lower=uv.min(axis=0)
            cols=int(math.ceil((uv[:,0].max()-lower[0])/5));rows=int(math.ceil((uv[:,1].max()-lower[1])/5))
            rr,cc=np.meshgrid(np.arange(rows),np.arange(cols),indexing='ij')
            grid_uv=np.c_[lower[0]+(cc.ravel()+.5)*5,lower[1]+(rr.ravel()+.5)*5]
            xy=grid_uv@matrix
            keep=Polygon(vv).contains_points(xy)
            coords=xy[keep];uvtasks=grid_uv[keep];rc=np.c_[rr.ravel(),cc.ravel()][keep]
            if (np.asarray(self.base)@matrix.T)[1]>=uvtasks[:,1].min()-1e-6:continue
            self.xy={i:tuple(p) for i,p in enumerate(coords)}
            self.uv={i:tuple(p) for i,p in enumerate(uvtasks)}
            self.rc={i:tuple(map(int,p)) for i,p in enumerate(rc)}
            self.rotation=matrix;self.byrow={}
            self.cache={}
            for i,(r,c) in self.rc.items():self.byrow.setdefault(r,[]).append(i)
            self.lower=lower
            route=self.route(set(self.xy),self.base,None,True)
            candidates.append((10*route['L']+50*route['A'],degrees,coords,uvtasks,rc,matrix,lower))
        _,self.degrees,coords,uvtasks,rc,self.rotation,self.lower=min(candidates,key=lambda a:(a[0],a[1]))
        self.xy={i:tuple(p) for i,p in enumerate(coords)};self.uv={i:tuple(p) for i,p in enumerate(uvtasks)}
        self.rc={i:tuple(map(int,p)) for i,p in enumerate(rc)};self.byrow={}
        self.cache={}
        for i,(r,c) in self.rc.items():self.byrow.setdefault(r,[]).append(i)
        self.all=set(self.xy)

    def feasible(self,job,start,heading=None):
        if not job:return True
        if tuple(start)==self.base and heading is None:return True
        su,sv=np.asarray(start)@self.rotation.T
        for k in job:
            u,v=self.uv[k]
            if v<sv-1e-7:return False
        equal=[self.uv[k][0]-su for k in job if abs(self.uv[k][1]-sv)<1e-7]
        if equal and min(equal)<-1e-7 and max(equal)>1e-7:return False
        if equal and heading is not None:
            h=np.asarray(heading)@self.rotation.T
            if abs(h[1])<1e-7 and h[0]*sum(equal)<-1e-7:return False
        return True

    def route(self,job,start,heading=None,launch=False):
        key=(frozenset(job),tuple(start),heading,launch)
        if key not in self.cache:self.cache[key]=self._route(job,start,heading,launch)
        return self.cache[key]

    def _route(self,job,start,heading=None,launch=False):
        if not job:return dict(ids=[],points=[tuple(start)],L=0.,A=0.)
        if not launch and not self.feasible(job,start,heading):return None
        groups={}
        for k in job:groups.setdefault(self.rc[k][0],[]).append(k)
        # Each state retains its endpoint and outgoing heading. The latter is
        # necessary when a row contains only a single task point.
        states=[(0.,tuple(start),heading,[],0.,0.)]
        for row in sorted(groups):
            ids=sorted(groups[row],key=lambda k:self.rc[k][1]);orders=[ids,ids[::-1]] if len(ids)>1 else [ids]
            nextstates={}
            for order in orders:
                a=self.xy[order[0]];b=self.xy[order[-1]]
                span=math.dist(a,b);rowhead=(b[0]-a[0],b[1]-a[1]) if len(order)>1 else None
                for cost,end,last,prev,l0,a0 in states:
                    connector=(a[0]-end[0],a[1]-end[1]);dist=math.hypot(*connector)
                    if dist<1e-9:
                        extra=angle(last,rowhead);out=rowhead or last
                    else:
                        if angle(last,connector)>1.999999 or angle(connector,rowhead)>1.999999:continue
                        extra=angle(last,connector)+angle(connector,rowhead);out=rowhead or connector
                    l1=l0+dist+span;a1=a0+extra;c1=10*l1+50*a1
                    key=(b,tuple(round(x,8) for x in out) if out is not None else None)
                    value=(c1,b,out,prev+order,l1,a1)
                    if key not in nextstates or (c1,tuple(value[3]))<(nextstates[key][0],tuple(nextstates[key][3])):nextstates[key]=value
            states=list(nextstates.values())
            if not states:return None
        best=min(states,key=lambda a:(a[0],tuple(a[3])))
        return dict(ids=best[3],points=[tuple(start)]+[self.xy[k] for k in best[3]],L=best[4],A=best[5])

    def initial_jobs(self,m):
        groups=sorted(self.byrow);counts=np.array([len(self.byrow[r]) for r in groups]);cumulative=counts.cumsum()
        # Keep whole rows together so every UAV has a distinct first row.
        cuts=[0]
        for i in range(1,m):
            cut=int(np.searchsorted(cumulative,i*cumulative[-1]/m))+1
            cut=max(cuts[-1]+1,min(cut,len(groups)-(m-i)));cuts.append(cut)
        cuts.append(len(groups))
        return [set(k for r in groups[cuts[i]:cuts[i+1]] for k in self.byrow[r]) for i in range(m)]

    def units(self,jobs,whole=False):
        units=[]
        for owner,job in enumerate(jobs):
            if whole:
                if job:units.append((owner,tuple(sorted(job))))
                continue
            for row in sorted({self.rc[k][0] for k in job}):
                ids=sorted((k for k in job if self.rc[k][0]==row),key=lambda k:self.rc[k][1]);piece=[];last=None
                for k in ids:
                    col=self.rc[k][1]
                    if piece and (col!=last+1 or len(piece)>=6):units.append((owner,tuple(piece)));piece=[]
                    piece.append(k);last=col
                if piece:units.append((owner,tuple(piece)))
        return units

def bcd_allocate(region,remaining,positions,headings,m):
    # Workload-balanced contiguous scan bands, adapted to the forward frontier.
    total=sum(map(len,remaining));jobs=[set() for _ in range(m)];unreachable=0
    for _,unit in region.units([set().union(*remaining)]):
        eligible=[i for i in range(m) if region.feasible(jobs[i]|set(unit),positions[i],headings[i])]
        if not eligible:
            owner=next(i for i,a in enumerate(remaining) if unit[0] in a);jobs[owner].update(unit);unreachable+=len(unit);continue
        i=min(eligible,key=lambda j:(abs((len(jobs[j])+len(unit))-total/m),len(jobs[j]),j))
        jobs[i].update(unit)
    return jobs,dict(unreachable=unreachable)

def ssi_allocate(region,remaining,positions,headings,m):
    jobs=[set() for _ in range(m)];plans=[region.route(set(),positions[i],headings[i]) for i in range(m)]
    blocks=list(region.units([set().union(*remaining)]));evals=0
    # Sequential single-item BIDSUMPATH: auctioneer announces blocks in scan
    # order. Each item goes to the robot with minimum incremental path bid.
    unassigned=[]
    for _,unit in blocks:
        best=None
        for i in range(m):
            route=region.route(jobs[i]|set(unit),positions[i],headings[i]);evals+=1
            if route is None:continue
            cand=(route['L']-plans[i]['L'],i,route)
            if best is None or cand[:2]<best[:2]:best=cand
        if best is None:unassigned.append(unit);continue
        _,i,route=best;jobs[i].update(unit);plans[i]=route
    for unit in unassigned:
        owner=next(i for i,a in enumerate(remaining) if unit[0] in a);jobs[owner].update(unit)
    return jobs,dict(candidates=evals,unreachable=sum(map(len,unassigned)))

def hql_allocate(region,remaining,positions,headings,q,battery,seed):
    """Reference-inspired two-stage tabular Q allocation, not author code.

    Global: block-to-UAV actions. Local: orientation-aware path DP acts as the
    deterministic geometry evaluator. The local orientation stage is NOT Q-
    learning; this adaptation is explicitly named QAlloc in the paper.
    """
    rng=np.random.default_rng(seed+803)
    blocks=region.units([set().union(*remaining)]);m=len(remaining)
    # Fixed task order makes the global state finite: block index + utilization bins.
    nominal=[];valid=[]
    for _,unit in blocks:
        costs=[];eligible=[]
        for i in range(m):
            route=region.route(set(unit),positions[i],headings[i]);eligible.append(route is not None)
            costs.append(demand(route['L'],route['A'],q[i])[0] if route else 1e9)
        nominal.append(costs);valid.append(eligible)
    Q={};best=None;episodes=96
    # Additive isolated-block bids are a surrogate; final path is recomputed.
    for ep in range(episodes):
        jobs=[set() for _ in range(m)];loads=np.zeros(m);trace=[]
        for k,(_,unit) in enumerate(blocks):
            options=[i for i in range(m) if valid[k][i]]
            if not options:options=[next(i for i,a in enumerate(remaining) if unit[0] in a)]
            state=(k,tuple(np.minimum(9,np.floor(loads/np.maximum(battery,1)*3)).astype(int)))
            values=Q.setdefault(state,np.zeros(m))
            epsilon=.25*(1-ep/episodes)+.02
            i=int(rng.choice(options)) if rng.random()<epsilon else min(options,key=lambda a:(-values[a],loads[a],a))
            before=loads[i];loads[i]+=nominal[k][i];jobs[i].update(unit)
            penalty=(loads[i]/max(battery[i],1))**2-(before/max(battery[i],1))**2
            reward=-nominal[k][i]/max(battery.sum(),1)-.25*penalty
            nxt=(k+1,tuple(np.minimum(9,np.floor(loads/np.maximum(battery,1)*3)).astype(int)))
            future=0 if k==len(blocks)-1 else max(Q.setdefault(nxt,np.zeros(m)))
            values[i]+=.3*(reward+.90*future-values[i])
        routes=[region.route(jobs[i],positions[i],headings[i]) for i in range(m)]
        if any(r is None for r in routes):continue
        ee=np.array([demand(r['L'],r['A'],q[i])[0] for i,r in enumerate(routes)])
        score=ee.sum()/max(battery.sum(),1)+2*np.maximum(ee/np.maximum(battery,1)-1,0).sum()
        if best is None or score<best[0]:best=(score,jobs)
    return (best[1] if best else [set(a) for a in remaining]),dict(episodes=episodes,q_states=len(Q))

def simulate(cfg,keep=False):
    tick=time.perf_counter();par=DEFAULT|cfg.get('par',{})
    geom,n,m,event,seed=cfg['geometry'],cfg['n'],cfg['m'],cfg['event'],cfg['seed']
    region=Region(geom,n,seed,par.get('rotation'));jobs=region.initial_jobs(m)
    init=[region.route(a,region.base,None,True) for a in jobs]
    assert all(r is not None for r in init)
    rng=np.random.default_rng(seed+geom*17011+123456)
    theta=np.exp(rng.normal(0,par['sigma'],8))[:m];q0=rng.uniform(.75,1.,8)[:m]
    eta=rng.normal(0,par['noise'],8)[:m];zeta=rng.normal(.08,par['drift'],8)[:m]
    initial=np.array([1.6*demand(r['L'],r['A'],q0[i])[0] for i,r in enumerate(init)])
    tevent=par['event_fraction']*max(demand(r['L'],r['A'],q0[i])[1] for i,r in enumerate(init))
    prefixes=[];observed=[];spent=[];headings=[];positions=[];seen=set();prefix_ids=[]
    for i,r in enumerate(init):
        points=[region.base];ids=[];e=0.;clock=0.;heading=None;base=0.
        for a,b,k in zip(r['points'],r['points'][1:],r['ids']):
            d=(b[0]-a[0],b[1]-a[1]);turn=angle(heading,d);length=math.hypot(*d)
            de,dt=demand(length,turn,q0[i],theta[i])
            if clock+dt>tevent or e+de>initial[i]:break
            e+=de;base+=de/theta[i];clock+=dt;points.append(b);ids.append(k);heading=d
        idle=max(0,tevent-clock);e+=theta[i]*20*idle;base+=20*idle
        prefixes.append(points);prefix_ids.append(ids);seen.update(ids);spent.append(e)
        observed.append(e*math.exp(eta[i])/base if base>0 else 1.)
        headings.append(heading);positions.append(points[-1])
    remaining=[a-seen for a in jobs];battery=initial-np.array(spent);q=q0.copy()
    damaged=m-1
    if event in (1,3,4):battery[damaged]*=par['battery_factor']
    if event in (2,3,4):q[m//2]=par['comm']
    truth_theta=theta*np.exp(zeta) if event==4 else theta.copy()
    refs=[region.route(a,positions[i],headings[i],positions[i]==region.base) for i,a in enumerate(remaining)]
    assert all(r is not None for r in refs)
    ref_e=np.array([demand(r['L'],r['A'],q[i])[0] for i,r in enumerate(refs)])
    ref_t=np.array([demand(r['L'],r['A'],q[i])[1] for i,r in enumerate(refs)])
    eref=max(1,ref_e.sum());tref=max(1,ref_t.max());kappa=math.exp(norm.ppf(1-par['delta']/m)*par['noise'])
    pre_ms=(time.perf_counter()-tick)*1000;out=[];details={}
    for method in cfg.get('methods',METHODS):
        t0=time.perf_counter();alloc=[set(a) for a in remaining];diag={};moves=0;evaluations=0;rejected=0;trace=[]
        factor=np.array(observed) if method in ('CF','NoRisk','NoComm','NoTurn','NoSplit','NoSafety') else np.ones(m)
        risk=1. if method=='NoRisk' else kappa
        if event and method=='BCD':alloc,diag=bcd_allocate(region,remaining,positions,headings,m)
        if event and method in ('SSI','SSI-E'):alloc,diag=ssi_allocate(region,remaining,positions,headings,m)
        if event and method=='QAlloc':alloc,diag=hql_allocate(region,remaining,positions,headings,q,battery,seed)
        plans=[region.route(a,positions[i],headings[i],positions[i]==region.base) for i,a in enumerate(alloc)]
        assert all(r is not None for r in plans)
        def evaluate(pp):
            et=[demand(r['L'],r['A'],q[i],factor[i],method=='NoTurn',method=='NoComm') for i,r in enumerate(pp)]
            ee=np.array([x[0] for x in et]);tt=np.array([x[1] for x in et]);util=ee/np.maximum(battery,1)
            phi=np.maximum(risk*ee/np.maximum((1-par['margin'])*battery,1)-1,0).sum()
            obj=.5*tt.max()/tref+.2*ee.sum()/eref+.15*tt.std()/tref+par['util_weight']*util.max()
            return ee,tt,float(phi),float(obj)
        if event and method in ('SSI-E','Energy','Model','CF','NoRisk','NoComm','NoTurn','NoSplit','NoSafety'):
            blocked=set();maxmoves=max(1,2*len(region.units(alloc)))
            for iteration in range(maxmoves):
                ee,tt,phi,obj=evaluate(plans);ratio=risk*ee/np.maximum((1-par['margin'])*battery,1)
                donors=[i for i in range(m) if ratio[i]>1+1e-9 and i not in blocked and alloc[i]]
                balancing=not donors
                if balancing:
                    if phi>1e-9 or method in ('SSI-E','Energy'):break
                    donor=int(np.argmax(tt))
                    if donor in blocked or not alloc[donor]:break
                else:donor=max(donors,key=lambda i:(ratio[i],i))
                units=region.units([alloc[donor]],method=='NoSplit');unit=set(units[-1][1])
                reduced=alloc[donor]-unit;route_d=region.route(reduced,positions[donor],headings[donor],positions[donor]==region.base)
                candidates=[]
                for receiver in range(m):
                    if receiver==donor:continue
                    route_r=region.route(alloc[receiver]|unit,positions[receiver],headings[receiver],positions[receiver]==region.base);evaluations+=1
                    if route_r is None:rejected+=1;continue
                    cand=plans.copy();cand[donor]=route_d;cand[receiver]=route_r
                    ce,ct,cp,cj=evaluate(cand)
                    if method!='NoSafety' and risk*ce[receiver]>(1-par['margin'])*battery[receiver]+1e-8:rejected+=1;continue
                    if balancing and (cp>1e-9 or cj>=obj-1e-9):continue
                    if not balancing and cp>=phi-1e-9:continue
                    score=(ce[receiver]-ee[receiver])/max(battery[receiver],1) if method in ('Energy','SSI-E') else cj
                    candidates.append((score,receiver,cand,cp,cj))
                if not candidates:blocked.add(donor);continue
                score,receiver,cand,cp,cj=min(candidates,key=lambda a:(a[0],a[1]))
                alloc[donor]-=unit;alloc[receiver]|=unit;plans=cand;moves+=1;blocked.clear()
                if keep:trace.append(dict(donor=donor,receiver=receiver,tasks=len(unit),phi_before=phi,phi_after=cp,J_before=obj,J_after=cj,phase='balance' if balancing else 'overload'))
        ee,tt,phi,obj=evaluate(plans);planning_ms=(time.perf_counter()-t0)*1000
        paths=[];visited=[];energy=[];times=[];actualL=0.;actualA=0.;required=[];stopped=0
        for i,route in enumerate(plans):
            ex=prefixes[i].copy();ids=prefix_ids[i].copy();use=0.;elapsed=0.;heading=headings[i]
            required.append(demand(route['L'],route['A'],q[i],truth_theta[i])[0])
            for a,b,k in zip(route['points'],route['points'][1:],route['ids']):
                d=(b[0]-a[0],b[1]-a[1]);ll=math.hypot(*d);aa=angle(heading,d);de,dt=demand(ll,aa,q[i],truth_theta[i])
                if use+de>battery[i]+1e-9:stopped+=1;break
                use+=de;elapsed+=dt;ex.append(b);ids.append(k);heading=d
            assert use<=battery[i]+1e-8
            paths.append(ex);visited.append(ids);energy.append(spent[i]+use);times.append(tevent+elapsed)
            ll,aa,_=path_stats(ex);actualL+=ll;actualA+=aa
        cov=len(set(k for ids in visited for k in ids))/len(region.all);true=np.array(required)
        no_repeat=sum(len(ids)-len(set(ids)) for ids in visited)
        assert no_repeat==0 and sum(map(len,visited))==len(set(k for ids in visited for k in ids))
        result=dict(suite=cfg.get('suite','core'),geometry=geom,event=event,n=n,side_m=n*5,m=m,seed=seed,method=method,
                    C=cov,success=int(cov>=.99),L=actualL,A=actualA,E=sum(energy),T_stop=max(times),stopped=stopped,
                    WAPE=float(np.abs(ee-true).sum()/max(true.sum(),1)),planning_ms=planning_ms,preprocessing_ms=pre_ms,
                    moves=moves,candidates=evaluations+diag.get('candidates',0),rejected=rejected,phi=phi,J=obj,
                    N=len(region.all),kappa=kappa,base_x=region.base[0],base_y=region.base[1],rotation=region.degrees,
                    repeat_tasks=no_repeat,factor=cfg.get('factor',''),level=cfg.get('level',''),episodes=diag.get('episodes',0))
        out.append(result)
        if keep:details[method]=dict(region=region,paths=paths,visited=visited,prefix=prefixes,prefix_ids=prefix_ids,plans=plans,alloc=alloc,
                                     initial=init,positions=positions,headings=headings,battery=battery,spent=spent,q=q,q0=q0,
                                     theta=theta,observed=observed,true_theta=truth_theta,event_t=tevent,trace=trace,result=result)
    return out,details

def configs(quick=False):
    out=[];reps=2 if quick else 8
    for g in range(1,7):
        for e in range(5):
            for n in (20,30,40):
                for m in (3,4,6,8):
                    for k in range(reps):out.append(dict(geometry=g,event=e,n=n,m=m,seed=BASE_SEED+k,suite='factorial'))
    if not quick:
        for g in range(1,7):
            for e in range(5):
                for k in range(8,40):out.append(dict(geometry=g,event=e,n=20,m=4,seed=BASE_SEED+k,suite='core_extra'))
    axes=dict(sigma=[.05,.20,.35],noise=[.01,.07,.15],battery_factor=[.15,.35,.55],comm=[.15,.35,.65],
              drift=[0,.15,.35],margin=[0,.05,.12],delta=[.01,.05,.15],util_weight=[0,.15,.30])
    for axis,levels in axes.items():
        for level in levels:
            for g in (1,3,6):
                for k in range(2 if quick else 12):
                    e=4 if axis=='drift' else 2 if axis=='comm' else 1
                    out.append(dict(geometry=g,event=e,n=20,m=4,seed=BASE_SEED+1000+k,suite='sensitivity',factor=axis,level=level,par={axis:level},methods=('Model','CF','Energy')))
    return out

def worker(cfg):return simulate(cfg)[0]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True);cc=configs(args.quick);t=time.perf_counter();allrows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,rows in enumerate(pool.map(worker,cc),1):
            allrows.extend(rows)
            if i%100==0:print(f'{i}/{len(cc)} states, {len(allrows)} runs, {time.perf_counter()-t:.1f}s',flush=True)
    with open(dest/'runs.csv','w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    config=dict(version='obstacle_free_fanout_v5',parameters=DEFAULT,base_seed=BASE_SEED,methods=METHODS,instances=len(cc),runs=len(allrows),quick=args.quick,
                shapes={k:v[0] for k,v in SHAPES.items()},events={0:'Static',1:'Battery',2:'Communication',3:'Combined',4:'Mechanism drift'},
                flight_rule='unrestricted obstacle-free airspace; camera off on transfers; strict sweep frontier after first task; no 180-degree reversal',
                python=os.sys.version)
    (dest/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Complete {len(allrows)} runs in {time.perf_counter()-t:.1f}s',flush=True)
