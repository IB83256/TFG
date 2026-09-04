import torch


import torch


def langevin_sampling(
    model,
    x_init,
    n_steps=100,
    step_size=0.01,
    noise_scale=0.01,
):
    x = x_init.clone().detach()

    trajectory = [x.clone()]

    for _ in range(n_steps):

        x.requires_grad_(True)

        energy = model(x).sum()

        grad = torch.autograd.grad(
            energy,
            x,
        )[0]

        with torch.no_grad():

            noise = noise_scale * torch.randn_like(x)

            x = x - step_size * grad + noise

        trajectory.append(x.clone())

    return torch.stack(trajectory)