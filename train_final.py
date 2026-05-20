"""PPO training with paper-accurate multi-slot environment."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path; from datetime import datetime
from envs.edge_offload_env import EdgeOffloadEnv
from agents.networks.actor_critic import ActorCriticNetwork

def discrete_to_dict(a, E):
    if a==0: return {"offload_ratio":np.array([0.0],np.float32),"target_es":0}
    return {"offload_ratio":np.array([1.0],np.float32),"target_es":min(a,E)}

def train(name, M, E, episodes, ep_len):
    device=torch.device('cpu')
    env=EdgeOffloadEnv(M,E,ep_len,3e9,10e9,500.0)
    dim,adim=env.obs_dim, E+1
    net=ActorCriticNetwork(dim,adim,128).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    
    hist=[];t0=time.time()
    for ep in range(episodes):
        obs,_=env.reset();traj={'s':[],'a':[],'r':[],'v':[],'lp':[],'d':[]}
        for step in range(ep_len):
            ob=np.array([obs[f"device_{i}"] for i in range(M)])
            st=torch.from_numpy(ob).float().to(device)
            with torch.no_grad():
                p,v=net(st);dist=torch.distributions.Categorical(p)
                acts=dist.sample();lp=dist.log_prob(acts)
            an=acts.numpy();ln=lp.numpy();vn=v.squeeze(-1).numpy()
            ad={f"device_{i}":discrete_to_dict(int(an[i]),E) for i in range(M)}
            no,rew,terms,_,_=env.step(ad)
            for i in range(M):
                a=f"device_{i}";traj['s'].append(obs[a]);traj['a'].append(an[i])
                traj['r'].append(rew[a]);traj['v'].append(vn[i]);traj['lp'].append(ln[i])
                traj['d'].append(terms[a])
            obs=no
            if any(terms.values()):break
        
        #PPO update
        n=len(traj['s'])
        if n>1:
            s=torch.from_numpy(np.array(traj['s'])).float().to(device)
            a=torch.from_numpy(np.array(traj['a'])).long().to(device)
            r=np.array(traj['r']);v=np.append(traj['v'],0.0);d=np.array(traj['d'],float)
            old_lp=torch.from_numpy(np.array(traj['lp'])).float().to(device)
            td=r+0.99*v[1:]*(1-d)-v[:-1];adv=np.zeros(n);g=0.0
            for t in reversed(range(n)):
                g=td[t]+0.99*0.95*(1-d[t])*g;adv[t]=g
            ret=adv+v[:-1];adv=(adv-adv.mean())/(adv.std()+1e-8)
            at=torch.from_numpy(adv).float().to(device)
            rt=torch.from_numpy(ret).float().to(device)
            for _ in range(4):
                perm=np.random.permutation(n)
                for i in range(0,n,64):
                    idx=perm[i:i+64];p2,v2=net(s[idx])
                    d2=torch.distributions.Categorical(p2)
                    nl=d2.log_prob(a[idx]);r2=torch.exp(nl-old_lp[idx])
                    s1=r2*at[idx];s2=torch.clamp(r2,0.8,1.2)*at[idx]
                    loss=-torch.min(s1,s2).mean()+0.5*(rt[idx]-v2.squeeze(-1)).pow(2).mean()-0.01*d2.entropy().mean()
                    opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),0.5);opt.step()
        
        if ep%200==0:
            m=env.get_episode_metrics();el=time.time()-t0
            hist.append({'ep':ep,**m,'time':el})
            print(f"  [{name}] ep{ep:5d} cost={m['avg_cost']:.2f} comp={m['completion_rate']:.3f} lat={m['avg_latency']:.1f} t={el:.0f}s")
    return hist

results={}
for cfg in [('IPPO_3MD',3,2,2000),('ExplabOff_3MD',3,2,2000),
            ('IPPO_5MD',5,3,2000),('ExplabOff_5MD',5,3,2000)]:
    name,M,E,eps=cfg
    print(f"\n{'='*60}\n{name} M={M} E={E} eps={eps}\n{'='*60}")
    results[name]=train(name,M,E,eps,100)

out=Path("results")/f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True,exist_ok=True)
with open(out/"results.json","w") as f:json.dump({k:[{kk:vv for kk,vv in v.items() if kk!='time'} for v in vs] for k,vs in results.items()},f,indent=2)
print(f"\nSaved to {out}")
for name,hist in results.items():
    if hist:print(f"  {name}: final cost={hist[-1]['avg_cost']:.2f} comp={hist[-1]['completion_rate']:.3f}")
