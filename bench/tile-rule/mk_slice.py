"""A 4-layer slice of the real Qwen3-30B-A3B checkpoint: same tensors for layers 0-3 + embed/norm/lm_head, symlinked shards."""
import json, os, sys
src, dst, L = "/models/Qwen3-30B-A3B", "/workspace/qwen3_l4_real", int(sys.argv[1]) if len(sys.argv) > 1 else 4
os.makedirs(dst, exist_ok=True)
cfg = json.load(open(f"{src}/config.json")); cfg["num_hidden_layers"] = L
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=1)
idx = json.load(open(f"{src}/model.safetensors.index.json"))
keep = {k: v for k, v in idx["weight_map"].items() if not k.startswith("model.layers.") or int(k.split(".")[2]) < L}
json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep}, open(f"{dst}/model.safetensors.index.json", "w"))
for f in set(keep.values()) | {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"}:
    p = f"{dst}/{f}"
    if not os.path.lexists(p):
        os.symlink(f"{src}/{f}", p)
print(dst, len(keep), "tensors over", len(set(keep.values())), "shards")
