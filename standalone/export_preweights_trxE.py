# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path: sys.path.insert(0, str(PROJ))
from algonauts2025.standalone.weighted_moe_decoder import FmriEncoder_MoE

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# --------- 单集按窗口读取（和你现有导出脚本一致风格）---------
def _load_LDT(path: Path):  # -> [L,D,T]
    arr = np.load(path); assert arr.ndim==3, arr.shape
    return np.transpose(arr, (1,2,0))

def _parse_layers_arg(s: str, L: int):
    s = (s or "").strip().lower()
    if not s: return "indices", [L-1]
    if s=="all": return "indices", list(range(L))
    if s.startswith("last"):
        try:k=int(s.replace("last",""))
        except:k=1
        k=max(1,min(k,L)); return "indices", list(range(L-k,L))
    if s.startswith("idx:"):
        idx=[]
        for p in [p for p in s[4:].split(",") if p.strip()]:
            try:i=int(p); 
            except:continue
            if 0<=i<L: idx.append(i)
        return "indices", sorted(set(idx or [L-1]))
    try:
        fr=[min(1,max(0,float(x))) for x in s.split(",") if x.strip()!=""]
        return "fractions", (fr or [1.0])
    except: return "indices",[L-1]

def _group_mean_layers(LDT: np.ndarray, fracs):
    L=LDT.shape[0]
    idxs=sorted(set(int(round(f*(L-1))) for f in fracs)) or [L-1]
    if idxs[-1]!=L-1: idxs[-1]=L-1
    bounds=[i+1 for i in idxs]; starts=[0]+bounds[:-1]
    out=[]
    for s,e in zip(starts,bounds):
        s=max(0,min(s,L)); e=max(0,min(e,L))
        if e<=s: s,e=L-1,L
        out.append(LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(out,axis=0)

class EpisodeWindows(Dataset):
    def __init__(self, episode_id, video_root, text_root, audio_root,
                 layers_arg, layer_agg, window_tr, stride_tr, frames_per_tr):
        self.ds=episode_id
        self.video_root, self.text_root, self.audio_root = map(Path, (video_root,text_root,audio_root))
        self.N=int(window_tr); self.S=int(stride_tr); self.f=int(frames_per_tr)
        v0=np.load(self.video_root/f"{self.ds}.npy")
        probe_L=v0.shape[1]
        self.mode, payload=_parse_layers_arg(layers_arg, probe_L)
        if self.mode=="fractions": self.fracs, self.sel=None, [float(x) for x in payload]
        else: self.fracs, self.sel=None, [int(i) for i in payload]
        self.layer_agg=layer_agg.lower()
        T_frames=v0.shape[0]; self.T_tr=T_frames//self.f
        self.index=[st for st in range(0,max(1,self.T_tr-self.N+1),self.S) if st+self.N<=self.T_tr]
        def _pick(LDT):
            L=LDT.shape[0]
            if self.mode=="indices":
                sel=[i for i in self.sel if 0<=i<L] or [L-1]; return LDT[sel]
            if self.layer_agg in ("group_mean","groupmean"): return _group_mean_layers(LDT, [float(x) for x in payload])
            sel=sorted(set(int(round(f*(L-1))) for f in [float(x) for x in payload])) or [L-1]
            sel=[min(L-1,max(0,i)) for i in sel]; return LDT[sel]
        vLDT=_load_LDT(self.video_root/f"{self.ds}.npy")
        tLDT=_load_LDT(self.text_root /f"{self.ds}.npy")
        aLDT=_load_LDT(self.audio_root/f"{self.ds}.npy")
        vG=_pick(vLDT); tG=_pick(tLDT); aG=_pick(aLDT)
        self.G,self.Dv=vG.shape[0],vG.shape[1]; self.Dt,self.Da=tG.shape[1],aG.shape[1]
        self._pick=_pick
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        st=self.index[i]; s_frame=st*self.f; e_frame=s_frame+self.N*self.f
        feats={}
        for name,root in (("video",self.video_root),("text",self.text_root),("audio",self.audio_root)):
            LDT=_load_LDT(root/f"{self.ds}.npy"); GDT=self._pick(LDT)
            if e_frame>GDT.shape[-1]: e_frame=GDT.shape[-1]; s_frame=e_frame-self.N*self.f
            feats[name]=torch.from_numpy(GDT[...,s_frame:e_frame].astype(np.float32))
        return {"video":feats["video"],"text":feats["text"],"audio":feats["audio"],
                "start_tr":int(st), "ds":self.ds}

def _collate(batch):
    out={k:torch.stack([b[k] for b in batch],0) for k in ["video","text","audio"]}
    out["start_tr_list"]=[b["start_tr"] for b in batch]; out["ds_list"]=[b["ds"] for b in batch]
    class B: 
        def __init__(self,d): self.data=d
        def to(self,dev):
            for k,v in self.data.items():
                if torch.is_tensor(v): self.data[k]=v.to(dev,non_blocking=True)
            return self
    return B(out)

@torch.no_grad()
def export_tr_e(model, episode, video_root, text_root, audio_root,
                layers, layer_agg, window_tr, stride_tr, frames_per_tr,
                device, subject_id, out_path: Path):
    ds=EpisodeWindows(episode, video_root, text_root, audio_root, layers, layer_agg,
                      window_tr, stride_tr, frames_per_tr)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=_collate,pin_memory=(device.type=='cuda'))
    E=model.num_experts; T=ds.T_tr
    sum_TE=np.zeros((T,E),dtype=np.float64); cnt_T=np.zeros((T,),dtype=np.int64)
    model.eval()
    for batch in loader:
        batch=batch.to(device)
        batch.data["subject_id"]=torch.full((1,),subject_id,dtype=torch.long,device=device)
        # forward_with_details -> (y, w_final, experts_out, w_pre)
        _,_,_,w_pre = model.forward_with_details(batch, pool_outputs=True)  # [1,N,E]
        w=w_pre[0].detach().cpu().numpy()  # [N,E]
        st=int(batch.data["start_tr_list"][0]); N=w.shape[0]; ed=min(st+N,T)
        sum_TE[st:ed]+=w[:ed-st]; cnt_T[st:ed]+=1
    cnt_T=np.maximum(cnt_T,1)
    mean_TE=(sum_TE/(cnt_T[:,None])).astype(np.float32)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.save(out_path, mean_TE)
    print(f"[EXPORT] saved {out_path}  shape={mean_TE.shape}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--episode_id",required=True)
    ap.add_argument("--video_root",required=True); ap.add_argument("--text_root",required=True); ap.add_argument("--audio_root",required=True)
    ap.add_argument("--layers",default="0.6,0.8,1.0"); ap.add_argument("--layer_aggregation",default="group_mean")
    ap.add_argument("--window_tr",type=int,default=100); ap.add_argument("--stride_tr",type=int,default=50); ap.add_argument("--frames_per_tr",type=int,default=3)
    ap.add_argument("--moe_num_experts",type=int,required=True); ap.add_argument("--moe_top_k",type=int,required=True)
    ap.add_argument("--moe_combine_mode",choices=["router","learned","router_x_learned"],required=True)
    ap.add_argument("--moe_subject_expert_bias",action="store_true"); ap.add_argument("--subject_embedding",action="store_true")
    ap.add_argument("--moe_dropout",type=float,default=0.1)
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--subject_id",type=int,default=0)
    ap.add_argument("--out_npy",required=True)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先用这一集推断 feature 维度
    ds=EpisodeWindows(args.episode_id, Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                      args.layers, args.layer_aggregation, args.window_tr, args.stride_tr, args.frames_per_tr)
    feat_dims={"video":(ds.G,ds.Dv),"text":(ds.G,ds.Dt),"audio":(ds.G,ds.Da)}
    model=FmriEncoder_MoE(feature_dims=feat_dims, n_outputs=1000, n_output_timesteps=args.window_tr,
                          n_subjects=4, num_experts=args.moe_num_experts, top_k=args.moe_top_k,
                          feature_aggregation="cat", layer_aggregation="cat",
                          subject_embedding=args.subject_embedding, moe_dropout=args.moe_dropout,
                          combine_mode=args.moe_combine_mode, subject_expert_bias=args.moe_subject_expert_bias).to(device)
    sd=torch.load(Path(args.checkpoint), map_location=device); model.load_state_dict(sd, strict=True)
    print("[LOAD] checkpoint OK")

    export_tr_e(model, args.episode_id, Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                args.layers, args.layer_aggregation, args.window_tr, args.stride_tr, args.frames_per_tr,
                device, args.subject_id, Path(args.out_npy))

if __name__=="__main__":
    main()