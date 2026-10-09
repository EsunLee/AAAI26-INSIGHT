import argparse,json,re
from pathlib import Path

ANS=re.compile(r'<answer>\s*(.*?)\s*</answer>',re.I|re.S)
ACT=re.compile(r'^[a-z0-9_:-]+\s+[a-z0-9_:-]+$',re.I)
BAD='<invalid> <invalid>'
PAD='<pad> <pad>'

def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def response(x):
    for k in ('response','prediction','generated_text','output','completion'):
        v=x.get(k)
        if isinstance(v,str): return v
        if isinstance(v,list) and v and isinstance(v[0],str): return v[0]
    a=[m.get('content','') for m in x.get('messages',[]) if m.get('role')=='assistant']
    return a[-1] if a else ''

def actions(s):
    m=ANS.search(s); body=m.group(1) if m else s
    out=[]
    for x in body.split(','):
        x=re.sub(r'^\s*(?:\d+[.)]|[-*])\s*','',x).strip().lower().strip(' .;')
        if x: out.append(x if ACT.fullmatch(x) else BAD)
    return out,bool(m)

def truth(x,h):
    a=x.get('future_actions')
    if not isinstance(a,list):
        a=actions(x.get('gt_answer',x.get('solution','')))[0]
    if len(a)<h: raise ValueError('invalid ground truth')
    return [str(y).strip().lower() for y in a[:h]]

def norm(a,h): return (a[:h]+[PAD]*h)[:h]

def dl(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0]=i
    for j in range(len(b)+1): d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            c=a[i-1]!=b[j-1]
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+c)
            if i>1 and j>1 and a[i-1]==b[j-2] and a[i-2]==b[j-1]:
                d[i][j]=min(d[i][j],d[i-2][j-2]+c)
    return d[-1][-1]/len(b)

def dist(a,b):
    av=[x.split(' ',1)[0] for x in a]; an=[x.split(' ',1)[1] for x in a]
    bv=[x.split(' ',1)[0] for x in b]; bn=[x.split(' ',1)[1] for x in b]
    return dl(a,b),dl(av,bv),dl(an,bn)

p=argparse.ArgumentParser()
p.add_argument('--ground-truth',required=True)
p.add_argument('--predictions',nargs='+',required=True)
p.add_argument('--output')
p.add_argument('--horizon',type=int,default=20)
a=p.parse_args(); gt=rows(a.ground_truth); ps=[rows(x) for x in a.predictions]
if any(len(x)!=len(gt) for x in ps): raise ValueError('row-count mismatch')
n=len(gt); top1=topk=fail=dup=tags=exact=valid=items=0
per=[0]*len(ps); one=[0.,0.,0.]; oracle=[0.,0.,0.]
for i,g in enumerate(gt):
    t=truth(g,a.horizon); first=[]; seq=[]
    for j,q in enumerate(ps):
        z,tag=actions(response(q[i])); f=z[0] if z and z[0]!=BAD else None
        first.append(f); seq.append(norm(z,a.horizon)); per[j]+=f==t[0]
        tags+=tag; exact+=len(z)==a.horizon; valid+=len(z)==a.horizon and BAD not in z; items+=len(z)
    top1+=first[0]==t[0]; topk+=t[0] in first; fail+=sum(x is None for x in first)
    good=[x for x in first if x]; dup+=len(good)-len(set(good))
    ds=[dist(x,t) for x in seq]
    for j in range(3): one[j]+=ds[0][j]; oracle[j]+=min(x[j] for x in ds)
k=len(ps); names=('action','verb','noun')
r={'samples':n,'candidate_files':k,
   'next_action_topk':{'top1':100*top1/n,f'top{k}':100*topk/n,'top1_correct':top1,f'top{k}_correct':topk,
      'per_file_top1':{Path(x).name:100*per[i]/n for i,x in enumerate(a.predictions)},
      'parse_failures':fail,'duplicate_candidate_slots':dup},
   'normalized_damerau_levenshtein':{'candidate1':{x:one[i]/n for i,x in enumerate(names)},
      f'oracle_min_over_{k}':{x:oracle[i]/n for i,x in enumerate(names)}},
   'generation_quality':{'answer_tag_rate':tags/(n*k),'exactly_20_items_rate':exact/(n*k),
      'exactly_20_valid_actions_rate':valid/(n*k),'mean_items':items/(n*k)}}
s=json.dumps(r,indent=2); print(s)
if a.output: Path(a.output).write_text(s+'\n')
