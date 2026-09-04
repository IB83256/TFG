import torch
from ebm.sampling.langevin import langevin_sampling
from torch.utils.data import DataLoader, TensorDataset

def train_energy_model(
    model,
    data,
    n_epochs=1000,
    batch_size=128,
    learning_rate=1e-3,
    langevin_steps=50,
    step_size=0.01,
    noise_scale=0.01,
):
    dataset = TensorDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    losses = []

    for epoch in range(n_epochs):

        for (real_samples,) in dataloader:

            x_init = 8 * torch.rand(
                real_samples.shape[0],
                2,
            ) - 2

            trajectory = langevin_sampling(
                model=model,
                x_init=x_init,
                n_steps=langevin_steps,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            model_samples = trajectory[-1].detach()

            real_energy = model(real_samples).mean()
            model_energy = model(model_samples).mean()

            loss = real_energy - model_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Loss: {loss.item():.4f} | "
                f"E_real: {real_energy.item():.4f} | "
                f"E_model: {model_energy.item():.4f}"
            )

    return losses


def train_energy_model_dsm(
    model,
    data,
    n_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    sigma=0.15,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    dataset = TensorDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    losses = []

    for epoch in range(n_epochs):

        epoch_loss = 0.0

        for (real_samples,) in dataloader:

            # Ruido gaussiano
            noise = torch.randn_like(real_samples)

            # Datos perturbados
            noisy_samples = (
                real_samples
                + sigma * noise
            )

            noisy_samples.requires_grad_(True)

            # Energía de los datos ruidosos
            energy = model(noisy_samples).sum()

            # Gradiente de la energía respecto a la entrada
            energy_gradient = torch.autograd.grad(
                energy,
                noisy_samples,
                create_graph=True,
            )[0]

            # Score del modelo:
            # s(x) = - grad_x E(x)
            model_score = -energy_gradient

            # Score objetivo del ruido gaussiano
            target_score = -noise / sigma

            # Pérdida DSM
            loss = (
                (model_score - target_score) ** 2
            ).sum(dim=1).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Loss DSM: {epoch_loss:.4f}"
            )

    return losses


def train_energy_model_pcd(
    model,
    data,
    n_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    langevin_steps=50,
    step_size=0.01,
    noise_scale=0.01,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    dataset = TensorDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Creamos una sola vez las cadenas persistentes
    persistent_samples = (
        8 * torch.rand(batch_size, 2) - 4
    )

    losses = []

    for epoch in range(n_epochs):

        epoch_loss = 0.0

        for (real_samples,) in dataloader:

            # Langevin empieza desde las muestras
            # finales del batch anterior
            trajectory = langevin_sampling(
                model=model,
                x_init=persistent_samples,
                n_steps=langevin_steps,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            model_samples = trajectory[-1].detach()

            # Guardamos las muestras finales
            # para utilizarlas en el siguiente batch
            persistent_samples = model_samples.clone()

            real_energy = model(
                real_samples
            ).mean()

            model_energy = model(
                model_samples
            ).mean()

            loss = (
                real_energy
                - model_energy
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)

        losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | "
                f"Loss PCD: {epoch_loss:.4f} | "
                f"E_real: {real_energy.item():.4f} | "
                f"E_model: {model_energy.item():.4f}"
            )

    return losses, persistent_samples



