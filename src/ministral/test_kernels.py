"""Independent NumPy checks of the native Vulkan kernels (no C++ runtime)."""
from pathlib import Path
import math,struct,subprocess
import numpy as np
root=Path(__file__).resolve().parent
out=root/'build'/'kernel-cases';out.mkdir(exist_ok=True)
rng=np.random.default_rng(7219)
def case(name,op,a,b,expected,groups,**kw):
 p=dict(op=op,indim=0,outdim=0,typ=0,tokens=1,start=0,maxseq=1,heads=1,kvheads=1,headdim=128,eps=1e-5,theta=1e6,freq=1/16,low=0.,high=1.,magnitude=1.,temp=.1,orig=16384);p.update(kw)
 push=struct.pack('<10I7fI',*[p[k] for k in ['op','indim','outdim','typ','tokens','start','maxseq','heads','kvheads','headdim','eps','theta','freq','low','high','magnitude','temp','orig']])
 arrays=[a.tobytes(),b.tobytes(),bytes(4),bytes(expected.size*4)]
 arrays=[x+bytes((-len(x))%4) for x in arrays]
 (out/(name+'.case')).write_bytes(struct.pack('<7I',*groups,*map(len,arrays))+push+b''.join(arrays))
 expected.astype('<f4').tofile(out/(name+'.expected'))
# Actual GGML layouts, with independent vectorized dequantization.
for kind,block,size in [(0,1,4),(1,1,2),(2,32,18),(3,32,20),(8,32,34),(14,256,210)]:
 width=512;rows=5;tokens=3;nb=width*rows//block
 if kind in (0,1):
  raw=rng.normal(size=(rows,width)).astype('<f4' if kind==0 else '<f2');w=raw.astype('float32')
 else:
  raw=rng.integers(0,256,(nb,size),dtype='uint8');ds=rng.uniform(.002,.04,nb).astype('<f2')
  if kind in (2,3,8):
   raw[:,:2]=ds.view('uint8').reshape(nb,2)
   if kind==3:
    mins=rng.uniform(-.5,.5,nb).astype('<f2');raw[:,2:4]=mins.view('uint8').reshape(nb,2)
   if kind==8:w=ds.astype('float32')[:,None]*raw[:,2:].view('int8')
   else:
    qs=raw[:,2 if kind==2 else 4:];q=np.concatenate([qs&15,qs>>4],axis=1).astype('float32')
    w=ds.astype('float32')[:,None]*(q-8) if kind==2 else ds.astype('float32')[:,None]*q+mins.astype('float32')[:,None]
  else:
   raw[:,208:210]=ds.view('uint8').reshape(nb,2)
   lo=raw[:,:128];hi=raw[:,128:192];sc=raw[:,192:208].view('int8');w=np.empty((nb,256),'float32')
   for half in range(2):
    ql=lo[:,half*64:(half+1)*64];qh=hi[:,half*32:(half+1)*32]
    for quad in range(4):
     q=(ql[:,(quad%2)*32:(quad%2+1)*32]>>(4*(quad//2)))&15
     q=q|(((qh>>(quad*2))&3)<<4)
     scale=np.repeat(sc[:,half*8+quad*2:half*8+quad*2+2],16,axis=1)
     w[:,half*128+quad*32:half*128+(quad+1)*32]=ds.astype('float32')[:,None]*scale*(q.astype('float32')-32)
  w=w.reshape(rows,width)
 x=rng.normal(size=(tokens,width)).astype('float32')
 case('matmul_'+str(kind),2,raw,x,x@w.T,(rows,tokens,1),indim=width,outdim=rows,typ=kind,tokens=tokens)
 ids=np.array([0,4,2],'<u4')
 case('embed_'+str(kind),0,raw,ids,w[ids],(3,1,1),indim=width,typ=kind)
x=rng.normal(size=(3,3072)).astype('float32');w=rng.normal(size=3072).astype('float32')
y=x/np.sqrt(np.mean(x.astype('float64')**2,axis=1)[:,None]+1e-5)*w
case('rmsnorm',1,x,w,y,(3,1,1),indim=3072)
for start in (0,32768):
 x=rng.normal(size=(3,2,128)).astype('float32');low=max(0,math.floor(128*math.log(16384/(32*2*math.pi))/(2*math.log(1e6))));high=min(127,math.ceil(128*math.log(16384/(2*math.pi))/(2*math.log(1e6))))
 angle=np.arange(start,start+3)[:,None]*np.power(1e6,-2*np.arange(64)/128)[None,:]
 mix=1-np.clip((np.arange(64)-low)/(high-low),0,1);angle*=mix+(1-mix)/16
 scale=1+.1*np.log(1+np.floor(np.arange(start,start+3)/16384))
 cs=(np.cos(angle)*scale[:,None])[:,None,:];sn=(np.sin(angle)*scale[:,None])[:,None,:]
 y=np.empty_like(x);y[:,:,::2]=x[:,:,::2]*cs-x[:,:,1::2]*sn;y[:,:,1::2]=x[:,:,::2]*sn+x[:,:,1::2]*cs
 f=np.power(1e6,-2*np.arange(64)/128)*(mix+(1-mix)/16);freqs=np.empty(128,'float32');freqs[::2]=f;freqs[1::2]=f-freqs[::2].astype('float64')
 case('rope_'+str(start),3,x,freqs,y,(2,3,1),heads=2,typ=1,start=start,low=low,high=high)
x=rng.normal(size=(2,512)).astype('float32');b=rng.normal(size=x.shape).astype('float32')
case('swiglu',8,x,b,x/(1+np.exp(-x))*b,(4,1,1),indim=512,tokens=2)
case('residual',9,x,b,x+b,(4,1,1),indim=512,tokens=2)
# 3 queries after 4 cached tokens, two GQA query heads for each KV head.
n=3;start=4;maxseq=7;heads=4;kv=2;hd=128
q=rng.normal(size=(n,heads,hd)).astype('float32');k=rng.normal(size=(maxseq,kv,hd)).astype('float32')
scores=np.einsum('thd,khd->thk',q,k.repeat(2,axis=1))/np.sqrt(hd)
for t in range(n):scores[t,:,start+t+1:]=-np.finfo('float32').max
case('scores',5,q,k,scores,(maxseq,n,heads),start=start,maxseq=maxseq,heads=heads,kvheads=kv)
probs=np.exp(scores-scores.max(axis=-1,keepdims=True));probs/=probs.sum(axis=-1,keepdims=True)
case('softmax',6,scores.astype('float32'),np.zeros(1,'float32'),probs,(heads,n,1),start=start,maxseq=maxseq,heads=heads)
v=rng.normal(size=(maxseq,kv,hd)).astype('float32');y=np.einsum('thk,khd->thd',probs,v.repeat(2,axis=1))
case('values',7,probs.astype('float32'),v,y,(heads,n,1),start=start,maxseq=maxseq,heads=heads,kvheads=kv)
subprocess.run([str(root/'build'/'MinistralKernelTest.exe'),str(out)],check=True)
fail=[]
for expected in sorted(out.glob('*.expected')):
 e=np.fromfile(expected,'<f4');a=np.fromfile(expected.with_suffix('.actual'),'<f4')
 # GLSL trigonometry has ordinary F32 argument-reduction error at long
 # positions.  The acceptance bound stays below 0.003 absolute for YaRN;
 # the remaining kernels use the tighter general bound.
 atol=3e-3 if expected.stem.startswith('rope_') else 3e-4
 ok=np.allclose(a,e,rtol=3e-4,atol=atol)
 print(expected.stem,'PASS' if ok else 'FAIL','max_abs',float(np.max(np.abs(a-e))))
 if not ok:fail.append(expected.stem)
assert not fail,fail
print('All',len(list(out.glob('*.expected'))),'Vulkan numerical cases passed')
