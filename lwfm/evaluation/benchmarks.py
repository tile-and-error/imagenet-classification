import io
import statistics
import time

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode


@torch.inference_mode()
def inference_cost(
    model: nn.Module, example: Tensor, *, warmup: int = 10, repeats: int = 50
) -> dict[str, float]:
    """Parameters, saved size, FLOPs and latency of `model` on `example`, run in
    eval mode on the example's device. Pass a batch of one for per-sample latency."""
    device = example.device
    was_training = model.training
    model.eval().to(device)

    buffer = io.BytesIO()  # saved size: the state_dict as torch.save writes it
    torch.save(model.state_dict(), buffer)
    with FlopCounterMode(display=False) as flops:  # one forward, counted
        model(example)

    # Latency: time each forward after a few warm-up runs (caches, kernel choice)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(
        device.type, lambda: None
    )
    times = []
    for i in range(warmup + repeats):
        sync()
        start = time.perf_counter()
        model(example)
        sync()  # GPU work is asynchronous; wait before stopping the clock
        if i >= warmup:
            times.append(time.perf_counter() - start)
    model.train(was_training)
    times.sort()
    return {
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "size_mb": buffer.getbuffer().nbytes / 1e6,
        "gflops": flops.get_total_flops() / 1e9,  # 2 FLOPs per multiply-add
        "latency_ms_median": 1e3 * statistics.median(times),
        "latency_ms_p95": 1e3 * times[int(0.95 * (len(times) - 1))],
    }
