"""Same bytes, two read patterns: the GEMV's strided-by-N vs fully coalesced."""
import statistics, sys, torch, triton, triton.language as tl

@triton.jit
def _strided(b_ptr, out_ptr, eids_ptr, K, N, sbe, sbn,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    g, pid_n = tl.program_id(0), tl.program_id(1)
    eid = tl.load(eids_ptr + g)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, 32)
    base = b_ptr + eid * sbe + offs_n[:, None] * sbn      # rows sbn apart
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        by = tl.load(base + ((k0 // 2) + offs_b)[None, :], mask=offs_n[:, None] < N, other=0)
        acc += tl.sum(by.to(tl.float32), axis=1)
    tl.store(out_ptr + g * N + pid_n, tl.sum(acc, axis=0))

@triton.jit
def _coalesced(b_ptr, out_ptr, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(1):
        by = tl.load(b_ptr + off, mask=off < n_elem, other=0)
        acc += by.to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))

dev="cuda"; E,N,K,T = 8,3072,4096,8
g=torch.Generator(device="cpu").manual_seed(0)
B=torch.randint(0,256,(E,N,K//2),dtype=torch.uint8,generator=g).to(dev)
eids=(torch.arange(T)%E).to(torch.int32).to(dev)
mb=B.numel()/1e6

def t(fn, mk, iters=7):
    for _ in range(8): mk()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s,e=torch.cuda.Event(True),torch.cuda.Event(True)
        s.record(); mk(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

out=torch.empty(T,N,dtype=torch.float32,device=dev)
grid=(T,triton.cdiv(N,64))
ms_s=t(_strided, lambda: _strided[grid](B,out,eids,K,N,B.stride(0),B.stride(1),BLOCK_N=64,BLOCK_K=64,num_warps=2,num_stages=3))
# coalesced over the SAME number of bytes the GEMV actually touches (T experts)
touched = T*N*(K//2)
flat = B.reshape(-1)[:touched].contiguous()
o2=torch.empty(triton.cdiv(touched,4096),dtype=torch.float32,device=dev)
gr2=(triton.cdiv(touched,4096),)
ms_c=t(_coalesced, lambda: _coalesced[gr2](flat,o2,touched,BLOCK=4096,num_warps=4,num_stages=3))
tmb=touched/1e6
print(f"  strided (GEMV pattern) {ms_s:.4f} ms  {tmb/1e3/(ms_s/1e3):6.1f} GB/s")
print(f"  coalesced (same bytes) {ms_c:.4f} ms  {tmb/1e3/(ms_c/1e3):6.1f} GB/s")
print(f"  ACCESS-PATTERN PENALTY: {ms_s/ms_c:.2f}x")
