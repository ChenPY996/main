from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolyPatch,FancyArrowPatch
from scipy.stats import norm
from simulation import simulate,Region,BASE_SEED,SHAPES,DEFAULT

HERE=Path(__file__).parent
RESULTS=HERE/'final_results' if (HERE/'final_results').exists() else HERE/'results';FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
T=json.load(open(RESULTS/'summary.json',encoding='utf-8'))
R=list(csv.DictReader(open(RESULTS/'runs.csv',encoding='utf-8-sig')))
for r in R:
 for k in r:
  if k in ('suite','method','factor','sweep_signs'):continue
  if r[k]!='':r[k]=float(r[k])
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
colors={'Fixed':'#777777','BCD':'#8172B2','SSI':'#64B5CD','SSI-E':'#C44E52','QAlloc':'#8C613C','Energy':'#55A868','Model':'#DD8452','CF':'#2B648D'}
uavcolors=['#2472A7','#E17D27','#3C9D50','#C64747','#8863AD','#259F9F','#C39421','#667682']
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)
def map_ax(ax,z):
 r=z['region'];ax.add_patch(PolyPatch(r.polygon,closed=True,facecolor='#F3F3F3',edgecolor='#555555',lw=.8))
 coords=np.array(list(r.xy.values()));ax.scatter(coords[:,0],coords[:,1],s=2,c='#C7C7C7',zorder=2)
 ax.scatter(*r.base,marker='*',s=42,c='black',zorder=9);ax.set_aspect('equal');ax.set(xlabel='x / m',ylabel='y / m');ax.grid(alpha=.08)
def path_ax(ax,z,title):
 map_ax(ax,z)
 for i,path in enumerate(z['paths']):
  points=np.array(path);col=uavcolors[i];ax.plot(points[:,0],points[:,1],color=col,lw=1.1,zorder=4,label=f'U{i}')
  ax.scatter(*points[-1],marker='x',s=19,c=col,zorder=5)
  if len(points)>1:
   a,b=points[0],points[1];ax.annotate('',xy=a+.8*(b-a),xytext=a+.6*(b-a),arrowprops=dict(arrowstyle='->',color=col,lw=.9))
 ax.set_title(title,fontsize=10)
 ax.legend(fontsize=7,ncol=2,loc='upper right',framealpha=.75)

# Fig 1: overview mechanism and information flow.
fig,ax=plt.subplots(figsize=(8.8,3.2));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
nodes={'geo':(1,3.2,'Polygon / sweep frame'),'x':(1,1.7,'State $X_k$'),'obs':(1,.45,'Energy telemetry'),
       'a':(4,3.2,'Allocation $a$'),'p':(4,1.7,'Forward route DP'),'th':(4,.45,r'$\hat{\theta}$ / bound'),
       'j':(7,1.7,r'Risk gate + $(\Phi,J)$'),'y':(9,1.7,'Execution')}
for x,y,label in nodes.values():ax.text(x,y,label,ha='center',va='center',bbox=dict(fc='white',ec='#333333',boxstyle='square,pad=.25'),fontsize=10)
for a,b in [('geo','p'),('x','p'),('a','p'),('obs','th'),('p','j'),('th','j'),('j','y')]:
 x1,y1=nodes[a][:2];x2,y2=nodes[b][:2];ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',shrinkA=35,shrinkB=38,mutation_scale=9,lw=.8))
save(fig,'mechanism.png')

# Fig 2: six separate geometry families, no holes.
fig,axs=plt.subplots(2,3,figsize=(8.5,5.5))
for g,ax in zip(range(1,7),axs.flat):
 r=Region(g,20,BASE_SEED);map_ax(ax,dict(region=r));ax.set_title(f'G{g}: {SHAPES[g][0]}\nrotation={r.degrees} deg',fontsize=10)
fig.tight_layout();save(fig,'geometries.png')

# Fig 3: DP geometry and a factual before/after transfer.
_,d=simulate(dict(geometry=3,event=1,n=20,m=4,seed=BASE_SEED,methods=('CF',)),True);z=d['CF']
fig,axs=plt.subplots(1,2,figsize=(8.6,3.5))
z0=z.copy();z0['paths']=[p['points'] for p in z['initial']]
path_ax(axs[0],z0,'(a) Nominal plan from the common departure point');path_ax(axs[1],z,'(b) Counterfactual reallocation')
for ax in axs:ax.scatter(*z['positions'][-1],marker='o',s=36,facecolors='none',edgecolors='black',zorder=8)
fig.tight_layout();save(fig,'reallocation.png')

# Fig 4: calibrated quantile and error-ranking bound.
fig,axs=plt.subplots(1,2,figsize=(8.4,2.6))
for m in (3,4,6,8):
 noise=np.linspace(0,.15,100);axs[0].plot(noise,np.exp(norm.ppf(1-.05/m)*noise),label=f'm={m}')
axs[0].set(xlabel='Telemetry log-noise',ylabel='Simultaneous multiplier');axs[0].legend(fontsize=8);axs[0].grid(alpha=.15)
eps=np.linspace(0,.15,100);axs[1].plot(eps,2*eps,color=colors['CF']);axs[1].set(xlabel=r'Uniform score error $\epsilon_J$',ylabel=r'Selection regret bound $2\epsilon_J$');axs[1].grid(alpha=.15)
fig.tight_layout();save(fig,'bounds.png')

# Fig 4b: mechanism diagnostic used by the results section.  The left panel
# shows the persistent individual factor recovered from the pre-event
# telemetry; the middle panel compares prediction error under the same
# factual states; the right panel reports the number of accepted transfers.
_,diag=simulate(dict(geometry=3,event=0,n=20,m=4,seed=BASE_SEED,methods=('CF',)),True)
dz=diag['CF']; theta=np.asarray(dz['theta']); observed=np.asarray(dz['observed'])
fig,axs=plt.subplots(1,3,figsize=(9.0,2.8))
axs[0].scatter(theta,observed,c=uavcolors[:len(theta)],s=38,edgecolor='white',linewidth=.4)
lo=min(theta.min(),observed.min())*.95;hi=max(theta.max(),observed.max())*1.05
axs[0].plot([lo,hi],[lo,hi],ls='--',color='#777777',lw=.9)
axs[0].set(xlabel=r'True $\theta_i$',ylabel=r'Telemetry estimate $\hat{\theta}_i$',title='Individual correction');axs[0].grid(alpha=.15)
for method,col in [('Model',colors['Model']),('CF',colors['CF']),('NoComm','#55A868'),('NoTurn','#C44E52')]:
 vals=[]
 for e in (1,2,3,4): vals.append(np.mean([r['WAPE']*100 for r in R if r['method']==method and r['event']==e and r['n']==20 and r['m']==4 and r['suite'] in ('factorial','core_extra')]))
 axs[1].plot([1,2,3,4],vals,'o-',label=method,color=col,lw=1.1,ms=3.5)
axs[1].set(xlabel='Event',ylabel='Energy WAPE (%)',title='Prediction under same facts');axs[1].set_xticks([1,2,3,4],['Battery','Comm','Joint','Drift'],rotation=20);axs[1].grid(alpha=.15);axs[1].legend(fontsize=7)
for method,col in [('Model',colors['Model']),('CF',colors['CF']),('NoSplit','#8172B2')]:
 vals=[np.mean([r['moves'] for r in R if r['method']==method and r['event']==e and r['n']==20 and r['m']==4 and r['suite'] in ('factorial','core_extra')]) for e in (1,2,3,4)]
 axs[2].plot([1,2,3,4],vals,'o-',label=method,color=col,lw=1.1,ms=3.5)
axs[2].set(xlabel='Event',ylabel='Accepted transfers',title='Counterfactual search');axs[2].set_xticks([1,2,3,4],['Battery','Comm','Joint','Drift'],rotation=20);axs[2].grid(alpha=.15);axs[2].legend(fontsize=7)
fig.tight_layout();save(fig,'mechanism_diagnostic.png')

# Fig 5: full plans in six geometries.
fig,axs=plt.subplots(2,3,figsize=(8.6,5.5))
for g,ax in zip(range(1,7),axs.flat):
 _,d=simulate(dict(geometry=g,event=0,n=20,m=4,seed=BASE_SEED,methods=('CF',)),True)
 path_ax(ax,d['CF'],f'G{g}: common departure point, four UAVs')
fig.tight_layout();save(fig,'paths_geometries.png')

# Fig 6: method trajectories in the same concave region.
fig,axs=plt.subplots(2,3,figsize=(8.6,5.5))
_,d=simulate(dict(geometry=3,event=3,n=20,m=4,seed=BASE_SEED,methods=('BCD','SSI-E','QAlloc','Energy','Model','CF')),True)
for ax,method in zip(axs.flat,d):path_ax(ax,d[method],f'{method}: C={100*d[method]["result"]["C"]:.1f}%')
fig.tight_layout();save(fig,'paths_methods.png')

# Fig 7: events with Wilson intervals.
fig,axs=plt.subplots(1,3,figsize=(9,3))
events=range(1,5)
for method in ('Fixed','BCD','SSI-E','QAlloc','Model','CF'):
 vals=[T['events'][str(e)][method] for e in events]
 for ax,key in zip(axs,('C','success','WAPE')):
  yy=[v[key] for v in vals];ax.plot(list(events),yy,'o-',label=method,color=colors[method],lw=1.2,ms=4)
  if key=='success':
   for e,v in zip(events,vals):
    p=v['success']/100;n=v['count'];den=1+1.96**2/n;ctr=(p+1.96**2/(2*n))/den;half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/den
    ax.vlines(e,100*(ctr-half),100*(ctr+half),color=colors[method],alpha=.35,lw=.8)
for ax,label in zip(axs,('Coverage (%)','Success rate (%)','Energy WAPE (%)')):
 ax.set(ylabel=label,xlabel='Event');ax.set_xticks(list(events),['Battery','Comm','Joint','Drift'],rotation=20);ax.grid(alpha=.15)
axs[0].legend(fontsize=8,ncol=2);fig.tight_layout();save(fig,'events.png')

# Fig 8: geometry/event heatmaps for CF.
fig,axs=plt.subplots(1,2,figsize=(8.5,3.4))
for ax,key,lab in zip(axs,('C','success'),('Coverage (%)','Success rate (%)')):
 data=[]
 for g in range(1,7):
  vals=[]
  for e in range(1,5):
   a=[r for r in R if r['geometry']==g and r['event']==e and r['method']=='CF' and r['n']==20 and r['m']==4 and r['suite'] in ('factorial','core_extra')]
   vals.append(np.mean([r[key] for r in a])*100)
  data.append(vals)
 im=ax.imshow(data,cmap='Blues',vmin=min(np.array(data).min(),80),vmax=100,aspect='auto')
 for g in range(6):
  for e in range(4):ax.text(e,g,f'{data[g][e]:.1f}',ha='center',va='center',color='white' if data[g][e]>94 else 'black',fontsize=10)
 ax.set_xticks(range(4),['Battery','Comm','Joint','Drift'],rotation=20);ax.set_yticks(range(6),[f'G{g}' for g in range(1,7)]);ax.set_title(lab);fig.colorbar(im,ax=ax,shrink=.85)
fig.tight_layout();save(fig,'heatmap.png')

# Fig 9: ablation effects, including the reserved-budget audit that justifies the
# risk gate: success alone cannot show what the screen is buying.
variants=('CF','Model','NoRisk','NoComm','NoTurn','NoSplit','NoSafety')
fig,axs=plt.subplots(1,4,figsize=(11.6,3.2))
for ax,key,label in zip(axs,('success','WAPE','T_stop','viol_reserve'),('Success rate (%)','Energy WAPE (%)','Stop time (s)','Reserve violation (%)')):
 yy=[T['pooled'][m][key] for m in variants];ax.bar(range(len(variants)),yy,color=[colors['CF']]+['#97A7B5']*6,width=.65)
 ax.set_xticks(range(len(variants)),variants,rotation=45,ha='right');ax.set(ylabel=label);ax.grid(axis='y',alpha=.15)
 for i,y in enumerate(yy):ax.text(i,y,f'{y:.1f}',ha='center',va='bottom',fontsize=8)
fig.tight_layout();save(fig,'ablation.png')

# Fig 15 (supplementary): measured first-leg fan-out separation, replacing the
# former unverified "all instances satisfy the fan-out criterion" statement.
F=T['fanout']
fig,axs=plt.subplots(1,2,figsize=(8.6,2.9))
rows=[r for r in R if r['method']=='CF' and r['suite'] in ('factorial','core_extra')]
seen={}
for r in rows:
 key=(r['geometry'],r['n'],r['m'],r['seed'])
 seen.setdefault(key,(r['fanout_min_deg'],r['fanout_min_perp'],r['fanout_min_gap']))
vals=np.array(list(seen.values()))
axs[0].hist(vals[:,0],bins=np.arange(0,16,1),color=colors['CF'],alpha=.85)
axs[0].axvline(1.0,color='#C44E52',ls='--',lw=1,label='enforced screen (1 deg)')
axs[0].set(xlabel='Minimum angle between launch legs (deg)',ylabel='Instances',title='First-leg fan-out separation');axs[0].legend(fontsize=8);axs[0].grid(alpha=.15)
axs[1].hist(vals[:,1],bins=np.arange(0,12,0.5),color='#55A868',alpha=.85)
axs[1].set(xlabel='Minimum perpendicular offset (m)',ylabel='Instances',title='Lateral separation of launch legs');axs[1].grid(alpha=.15)
fig.tight_layout();save(fig,'fanout.png')

# Fig 10: scales.  (a)-(c) area size x fleet size for CF; (d)-(f) the fleet-size
# sweep with seven methods (3 to 12 UAVs on the 100 m instances).
fig,axs=plt.subplots(2,3,figsize=(10.2,5.6))
top=(axs[0,0],axs[0,1],axs[0,2])
for n in (20,30,40):
 for ax,key in zip(top,('C','T_stop','planning_median')):
  selected=[r for r in T['scale'] if r['n']==n];ax.plot([r['m'] for r in selected],[r[key] for r in selected],'o-',label=f'{n*5} m',ms=4)
for ax,label in zip(top,('Coverage (%)','Stop time (s)','Planner median (ms)')):
 ax.set(xlabel='UAVs',ylabel=label);ax.set_xticks([3,4,6,8]);ax.grid(alpha=.15)
top[0].legend(fontsize=7.5);top[0].set_title('(a)-(c) CF vs area and fleet size',fontsize=9)
SWM=('CF','Model','Energy','QAlloc','SSI-E','NoComm','NoSafety')
SWC={'CF':colors['CF'],'Model':colors['Model'],'Energy':colors['Energy'],'QAlloc':colors['QAlloc'],
     'SSI-E':colors['SSI-E'],'NoComm':'#55A868','NoSafety':'#C44E52'}
if 'swarm' in T:
 wkeys=sorted(T['swarm'],key=int);sizes=[int(k) for k in wkeys]
 bot=(axs[1,0],axs[1,1],axs[1,2])
 for meth in SWM:
  bot[0].plot(sizes,[T['swarm'][k][meth]['success'] for k in wkeys],'o-',color=SWC[meth],label=meth,ms=3.5,lw=1.1)
  bot[1].plot(sizes,[T['swarm'][k][meth]['viol_battery'] for k in wkeys],'s-',color=SWC[meth],label=meth,ms=3,lw=1.0)
  bot[2].plot(sizes,[T['swarm'][k][meth]['planning_ms'] for k in wkeys],'^-',color=SWC[meth],label=meth,ms=3,lw=1.0)
 bot[0].set(xlabel='UAVs',ylabel='Success rate (%)',title='(d) Fleet-size sweep: success');bot[0].legend(fontsize=6.5,ncol=2)
 bot[1].set(xlabel='UAVs',ylabel='Battery-overrun rate (%)',title='(e) Fleet-size sweep: overrun');bot[1].legend(fontsize=6.5,ncol=2)
 bot[2].set(xlabel='UAVs',ylabel='Planner median (ms)',title='(f) Fleet-size sweep: decision cost');bot[2].legend(fontsize=6.5,ncol=2)
 bot[2].set_yscale('log')
 for ax in bot:ax.set_xticks([3,4,5,6,8,10,12]);ax.grid(alpha=.15)
fig.tight_layout();save(fig,'scale.png')

# Fig 11: parameter sensitivities.
factors=[('sigma','Heterogeneity'),('noise','Telemetry noise'),('battery_factor','Battery factor'),('drift','Mechanism drift'),('comm','Communication quality'),('margin','Energy reserve'),('delta','Risk probability'),('util_weight','Utilization weight')]
fig,axs=plt.subplots(2,4,figsize=(10,4.6))
for ax,(factor,label) in zip(axs.flat,factors):
 levels=sorted(T['sensitivity'][factor],key=float)
 for method in ('Model','CF','Energy'):
  ax.plot([float(x) for x in levels],[T['sensitivity'][factor][x][method]['C'] for x in levels],'o-',color=colors[method],label=method,ms=3,lw=1)
 ax.set_xticks([float(x) for x in levels]);ax.set_xticklabels([('%g' % float(x)) for x in levels],fontsize=7.5)
 ax.set(xlabel=label,ylabel='Coverage (%)');ax.grid(alpha=.15);ax.margins(x=.12)
axs[0,0].legend(fontsize=8);fig.tight_layout();save(fig,'sensitivity.png')

# Fig 12: paired uncertainty, all four events.
fig,axs=plt.subplots(1,3,figsize=(8.8,3))
for ax,metric,label in zip(axs,('C','T_stop','WAPE'),('Coverage difference (pp)','Stop-time difference (s)','WAPE difference (pp)')):
 rows=[r for r in T['model_paired'] if r['metric']==metric]
 ax.errorbar([r['event'] for r in rows],[r['delta'] for r in rows],yerr=[[r['delta']-r['low'] for r in rows],[r['high']-r['delta'] for r in rows]],fmt='o',color=colors['CF'],capsize=3)
 ax.axhline(0,color='#777777',ls='--',lw=.8);ax.set(ylabel=label,xlabel='Event');ax.set_xticks([1,2,3,4],['Battery','Comm','Joint','Drift'],rotation=20);ax.grid(alpha=.15)
fig.tight_layout();save(fig,'paired.png')

# Fig 16 (supplementary analysis): safety-performance trade-off, planning cost, the
# activation range of the risk screen, and the cost of crossing a concave notch.
import json as _json
SUP=_json.load(open(RESULTS/'supplementary.json',encoding='utf-8'))
NRY=_json.load(open(RESULTS/'notch.json',encoding='utf-8')) if (RESULTS/'notch.json').exists() else []
fig,axs=plt.subplots(2,2,figsize=(9.4,6.0))
ax=axs[0,0]
for m in colors:
 t=SUP['methods'][m]
 safe=100-T['pooled'][m]['viol_battery'] if False else 100*(T['pooled'][m]['success']/100)*(1-T['pooled'][m]['viol_battery']/100)
 ax.scatter(t['viol_battery'],t['success'],s=42,color=colors[m],zorder=4)
 ax.annotate(m,(t['viol_battery'],t['success']),textcoords='offset points',xytext=(4,-3),fontsize=7.5)
 ax.scatter(t['viol_battery'],safe,marker='^',s=24,facecolors='none',edgecolors=colors[m],zorder=4)
ax.set(xlabel='Battery-overrun rate (%)',ylabel='Success / safe-success (%)',title='(a) Safety-performance trade-off');ax.grid(alpha=.15)
ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color='#555555',label='success'),
                   plt.Line2D([],[],marker='^',ls='',mfc='none',color='#555555',label='success without overrun')],fontsize=7.5)
ax=axs[0,1]
meths=[m for m in ('Fixed','BCD','SSI','SSI-E','QAlloc','Energy','Model','CF','NoRisk','NoComm','NoTurn','NoSplit','NoSafety')]
med=[SUP['methods'][m]['plan_med'] for m in meths];p95=[SUP['methods'][m]['plan_p95'] for m in meths]
ax.bar(range(len(meths)),med,color=[colors.get(m,'#999999') for m in meths],width=.6)
ax.errorbar(range(len(meths)),med,yerr=[[0]*len(meths),[p-m for p,m in zip(p95,med)]],fmt='none',ecolor='#555555',capsize=2,lw=.8)
ax.set_xticks(range(len(meths)),meths,rotation=45,ha='right',fontsize=8);ax.set(ylabel='Planning time (ms)',title='(b) On-line decision cost (median, p95)');ax.grid(axis='y',alpha=.15)
ax=axs[1,0]
noises=sorted(float(k) for k in SUP['risk_gate'])
for m,c in (('CF',colors['CF']),('NoRisk',colors['NoRisk'] if 'NoRisk' in colors else '#8C613C'),('NoSafety','#C44E52'),('NoComm','#55A868')):
 ax.plot(noises,[SUP['risk_gate'][str(n)][m]['success'] for n in noises],'o-',color=c,label=m,ms=3.5,lw=1.1)
 ax.plot(noises,[SUP['risk_gate'][str(n)][m]['viol_battery'] for n in noises],'s--',color=c,ms=3,lw=.9,alpha=.75)
ax.axvline(DEFAULT['noise'],color='#777777',ls=':',lw=1)
ax.set(xlabel='Telemetry log-noise sigma',ylabel='Success (solid) / overrun (dashed), %',title='(c) Activation range of the risk screen');ax.legend(fontsize=7.5);ax.grid(alpha=.15)
ax=axs[1,1]
if NRY:
 lbl=['G%d U%d'%(r['geometry'],r['uav']) for r in NRY]
 x=np.arange(len(NRY))
 ax.bar(x-.2,[r['direct'] for r in NRY],width=.4,color=colors['CF'],label='straight across the gap')
 ax.bar(x+.2,[r['detour'] for r in NRY],width=.4,color='#97A7B5',label='detour inside the region')
 ax.set_xticks(x,lbl,rotation=60,ha='right',fontsize=7)
ax.set(ylabel='Segment length (m)',title='(d) Cost of crossing a concave notch');ax.legend(fontsize=7.5);ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'analysis.png')
print(FIG)



