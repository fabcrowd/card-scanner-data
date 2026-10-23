"""Rerun arm B exactly as train_pilot.py --arm B --raw-final, additionally saving the raw final student state_dict."""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_pilot as tp  # noqa: E402

_orig = tp.evaluate


def evaluate(ctx, model, teacher, ema_like, device, stem, epoch):
    if os.path.basename(stem) == "raw-final":
        torch.save({k: v.detach().cpu() for k, v in ema_like.state_dict().items()}, stem + ".pt")
    return _orig(ctx, model, teacher, ema_like, device, stem, epoch)


tp.evaluate = evaluate
if __name__ == "__main__":
    sys.argv = ["train_pilot.py", "--arm", "B", "--raw-final"]
    tp.main()
