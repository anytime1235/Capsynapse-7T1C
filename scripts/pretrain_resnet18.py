"""Software pretraining of ResNet-18 on CIFAR-10 (40 epochs, OneCycle SGD, bf16 autocast).

  python scripts/pretrain_resnet18.py [epochs] [output]

data/ckpt/resnet18_cifar10.pt (94.25 %) was trained with this script.
Default output: data/ckpt/resnet18_cifar10_new.pt.
"""
import os, sys, time
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from varsim.nn.resnet_cifar import resnet18_cifar, loaders, evaluate

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data/ckpt/resnet18_cifar10_new.pt")
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists")
torch.manual_seed(0); dev = "cuda"
tr, te = loaders(root=os.path.join(ROOT, "data/cifar"))
m = resnet18_cifar().to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, epochs=EPOCHS, steps_per_epoch=len(tr))
scaler = torch.amp.GradScaler(); crit = nn.CrossEntropyLoss()
best = 0
for ep in range(EPOCHS):
    m.train(); t = time.time()
    for x, y in tr:
        x, y = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(m(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    acc = evaluate(m, te)
    if acc > best:
        best = acc; torch.save(m.state_dict(), OUT)
    print(f"epoch {ep+1}/{EPOCHS} loss {loss.item():.3f} test acc {acc:.2f} (best {best:.2f}) {time.time()-t:.0f}s", flush=True)
