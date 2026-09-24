"""Numerical Vulkan checks against NumPy and gguf's independent dequantizer.
Run after building Qwen35KernelTest.exe: python test_kernels.py
"""
import struct, subprocess
from pathlib import Path
import numpy as np
from gguf import GGMLQuantizationType, dequantize
R = np.random.default_rng(3508)
ROOT = Path(__file__).resolve().parent
CASES = ROOT/'build'/'cases'
CASES.mkdir(exist_ok=True)
expected = {}
def case(name,op,a,b,c,out,groups=(1,1,1),state=None,expect_state=None,**kw):
    ints = dict(op=op,inDim=128,outDim=0,typ=0,tokens=1,startPos=0,maxSeq=64,heads=16,kvHeads=2,headDim=256)
    ints.update({k:v for k,v in kw.items() if k in ints})
    floats = dict(eps=1e-6,theta=1e7,freqScale=1.,corrLow=0.,corrHigh=0.,magnitude=1.,tempScale=0.)
    floats.update({k:v for k,v in kw.items() if k in floats})
    push = struct.pack('<10I7fI',*ints.values(),*floats.values(),0)
    def raw(x):return np.asarray(x,np.float32).tobytes() if not isinstance(x,bytes) else x
    data = [raw(a),raw(b),raw(c),np.zeros_like(out,dtype=np.float32).tobytes(),raw([0] if state is None else state)]
    data = [x+b'\0'*((-len(x))%4) for x in data]
    (CASES/(name+'.case')).write_bytes(struct.pack('<3I5I',*groups,*map(len,data))+push+b''.join(data))
    expected[name]=(np.asarray(out,np.float32).ravel(),expect_state)
def norm(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
def silu(x):return x/(1+np.exp(-x))
# Quantized layouts use random packed blocks, including nontrivial scale high bits.
for typ,bs in [(12,144),(14,210)]:
    raw=R.integers(0,256,(3,bs),dtype=np.uint8)
    if typ==12:
        raw[:,:4]=np.tile(np.array([.02,.01],np.float16).view(np.uint8),(3,1))
    else:raw[:,-2:]=np.tile(np.array([.002],np.float16).view(np.uint8),(3,1))
    w=dequantize(raw,GGMLQuantizationType(typ)).reshape(3,256)
    x=R.normal(size=256).astype(np.float32)
    case('matvec_'+str(typ),2,raw.tobytes(),x,[0],w@x,groups=(3,1,1),inDim=256,outDim=3,typ=typ)
# Conv history at fresh and continuing positions
for pos in [0,7]:
    a=R.normal(size=6144).astype(np.float32); w=R.normal(size=(6144,4)).astype(np.float32)
    hist=R.normal(size=(6144,3)).astype(np.float32)
    values=np.column_stack([np.zeros_like(hist) if pos==0 else hist,a])
    case('conv_'+str(pos),10,a,w,[0],silu((values*w).sum(1)),groups=(24,1,1),state=hist,
         expect_state=values[:,1:],inDim=6144,startPos=pos)
x=R.normal(size=(32,128)).astype(np.float32)
case('l2',11,x,[0],[0],x/np.sqrt((x*x).sum(1,keepdims=True)+1e-6),groups=(32,1,1))
a=R.normal(size=16).astype(np.float32);b=R.normal(size=16).astype(np.float32);c=-np.exp(R.normal(size=16)).astype(np.float32)
case('decay',12,a,b,c,np.exp(c*np.logaddexp(0,a+b)),inDim=16)
# Full independent outer-product recurrence, repeated for reset and nonzero state.
for pos in [0,13]:
    q,k,v=[R.normal(size=(16,128)).astype(np.float32) for _ in range(3)]
    q/=np.sqrt((q*q).sum(-1,keepdims=True)+1e-6); k/=np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
    decay=R.uniform(.1,.99,size=16).astype(np.float32); beta=R.normal(size=16).astype(np.float32)
    state=R.normal(size=(16,128,128)).astype(np.float32) # h,value,key
    old=(state if pos else np.zeros_like(state))*decay[:,None,None]
    delta=(v-np.einsum('hvk,hk->hv',old,k))/(1+np.exp(-beta[:,None]))
    updated=old+delta[:,:,None]*k[:,None,:]
    out=np.einsum('hvk,hk->hv',updated,q)/np.sqrt(128)
    case('delta_'+str(pos),13,np.stack([q,k,v]),decay,beta,out,groups=(128,16,1),state=state,
         expect_state=updated,startPos=pos)
x=R.normal(size=(16,128)).astype(np.float32);w=R.normal(size=128).astype(np.float32);z=R.normal(size=x.shape).astype(np.float32)
case('gated_norm',14,x,w,z,norm(x)*w*silu(z),groups=(16,1,1))
x=R.normal(size=(8,512)).astype(np.float32);w=R.normal(size=256).astype(np.float32)
case('q_split_norm',15,x,w,[0],norm(x[:,:256])*w,groups=(8,1,1))
# RoPE op writes only first 64 values; remaining values use zero initial output in this harness.
x=R.normal(size=(8,256)).astype(np.float32);out=np.zeros_like(x);angle=123*np.power(1e7,-np.arange(32)*2/64)
out[:,:32]=x[:,:32]*np.cos(angle)-x[:,32:64]*np.sin(angle)
out[:,32:64]=x[:,:32]*np.sin(angle)+x[:,32:64]*np.cos(angle)
case('rope',16,x,[0],[0],out,groups=(8,1,1),startPos=123,outDim=64)
y=R.normal(size=(8,512)).astype(np.float32)
case('attention_gate',17,x,y,[0],x/(1+np.exp(-y[:,256:])),groups=(8,1,1),inDim=2048)
subprocess.run([str(ROOT/'build'/'Qwen35KernelTest.exe'),str(CASES)],check=True,capture_output=True)
for name,(out,state) in expected.items():
    actual=np.fromfile(CASES/(name+'.actual'),np.float32)
    np.testing.assert_allclose(actual,out,rtol=3e-4,atol=3e-4,err_msg=name)
    if state is not None:
        actual_state=np.fromfile(CASES/(name+'.state'),np.float32)
        np.testing.assert_allclose(actual_state,state.ravel(),rtol=3e-4,atol=3e-4,err_msg=name+' state')
    print(name,'PASS')
print(len(expected),'numerical checks passed')
