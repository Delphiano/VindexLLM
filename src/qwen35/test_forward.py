"""Compare every decoder layer over three tokens with independent NumPy math."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys
import numpy as np
from gguf import GGUFReader,dequantize
r=GGUFReader(sys.argv[1]);t={x.name:x for x in r.tensors}
def w(name):return dequantize(t[name].data,t[name].tensor_type).astype(np.float32)
def norm(x,weight):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*weight
def silu(x):return x/(1+np.exp(-np.clip(x,-80,80)))
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-80,80)))
def rope(x,pos):
    y=x.copy(); a=pos*np.power(1e7,-np.arange(32,dtype=np.float32)*2/64)
    y[...,:32]=x[...,:32]*np.cos(a)-x[...,32:64]*np.sin(a)
    y[...,32:64]=x[...,:32]*np.sin(a)+x[...,32:64]*np.cos(a)
    return y
ids=[9419,1814,0]
e=t['token_embd.weight'];x=dequantize(e.data[ids],e.tensor_type).astype(np.float32)
actual=np.fromfile('build/forward.bin',np.float32).reshape(3,24,1024)
for layer in range(24):
    p=f'blk.{layer}.'; inp=norm(x,w(p+'attn_norm.weight'))
    if (layer+1)%4:
        qkv=inp@w(p+'attn_qkv.weight').T
        z=(inp@w(p+'attn_gate.weight').T).reshape(3,16,128)
        alpha=inp@w(p+'ssm_alpha.weight').T
        beta=sigmoid(inp@w(p+'ssm_beta.weight').T)
        decay=np.exp(w(p+'ssm_a')*np.logaddexp(0,alpha+w(p+'ssm_dt.bias')))
        conv=w(p+'ssm_conv1d.weight');hist=np.zeros((6144,3),np.float32)
        state=np.zeros((16,128,128),np.float32);out=[]
        for pos in range(3):
            vals=np.column_stack((hist,qkv[pos]));hist=vals[:,1:]
            q,k,v=silu((vals*conv).sum(1)).reshape(3,16,128)
            q=q/np.sqrt((q*q).sum(-1,keepdims=True)+1e-6)/np.sqrt(128)
            k=k/np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
            state=state*decay[pos,:,None,None]
            delta=(v-np.einsum('hkv,hk->hv',state,k))*beta[pos,:,None]
            state+=k[:,:,None]*delta[:,None,:]
            out.append(np.einsum('hkv,hk->hv',state,q))
        out=norm(np.array(out),w(p+'ssm_norm.weight'))*silu(z)
        attn=out.reshape(3,2048)@w(p+'ssm_out.weight').T
    else:
        qg=(inp@w(p+'attn_q.weight').T).reshape(3,8,512)
        q=norm(qg[:,:,:256],w(p+'attn_q_norm.weight'))
        k=norm((inp@w(p+'attn_k.weight').T).reshape(3,2,256),w(p+'attn_k_norm.weight'))
        v=(inp@w(p+'attn_v.weight').T).reshape(3,2,256)
        q=np.array([rope(q[i],i) for i in range(3)]);k=np.array([rope(k[i],i) for i in range(3)])
        out=np.zeros((3,8,256),np.float32)
        for pos in range(3):
            for h in range(8):
                score=k[:pos+1,h//4]@q[pos,h]/16
                prob=np.exp(score-score.max());prob/=prob.sum()
                out[pos,h]=prob@v[:pos+1,h//4]
        out*=sigmoid(qg[:,:,256:])
        attn=out.reshape(3,2048)@w(p+'attn_output.weight').T
    x=x+attn
    inp=norm(x,w(p+'post_attention_norm.weight'))
    x=x+(silu(inp@w(p+'ffn_gate.weight').T)*(inp@w(p+'ffn_up.weight').T))@w(p+'ffn_down.weight').T
    err=np.max(np.abs(x-actual[:,layer])); rel=np.linalg.norm(x-actual[:,layer])/np.linalg.norm(x)
    print('layer',layer,'max_abs',round(float(err),6),'relative',round(float(rel),7),flush=True)
    assert rel<.001, f'layer {layer} differs'
print('All 24 layers match the independent reference across 3 recurrent steps.')
