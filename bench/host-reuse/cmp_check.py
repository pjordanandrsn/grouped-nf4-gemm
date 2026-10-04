import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
assert a.keys() == b.keys(), (a.keys(), b.keys())
bad = [k for k in a if not torch.equal(a[k], b[k])]
print("tensors", len(a), "bitwise-different:", bad or "none")
