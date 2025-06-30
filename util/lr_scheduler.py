import numpy as np
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    # print(base_value, final_value, epochs, niter_per_ep, warmup_epochs, start_warmup_value, warmup_steps)
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    # print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def linear_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.zeros(epochs * niter_per_ep)
    total_iters = epochs * niter_per_ep
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            schedule[epoch * niter_per_ep : (epoch + 1) * niter_per_ep] = np.linspace(start_warmup_value, base_value, niter_per_ep)
        else:
            t_max = total_iters - warmup_iters
            for i in range(niter_per_ep):
                t = epoch * niter_per_ep - warmup_iters + i
                schedule[epoch * niter_per_ep + i] = base_value - (base_value - final_value) * t / t_max

    return schedule
