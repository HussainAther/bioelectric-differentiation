import time
import torch
import torch.nn.functional as F

# Parameters
GRID_SIZE = (30, 30, 10)  # (Depth/X, Height/Y, Width/Z)
STEPS = 1000
DIFFUSION_RATE = 0.2
DECAY_RATE = 0.05
STIMULUS_STRENGTH = 1.0
THRESHOLD = 0.6
KT = 0.05
COUPLING_STRENGTH = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running 3D Bioelectric Simulation on: {torch.cuda.get_device_name(0)}")

# 1. Initialize Grids on GPU VRAM
voltage = (torch.rand(1, 1, *GRID_SIZE, device=DEVICE) * 0.1)  # Shape: (1, 1, 30, 30, 10)
states = torch.zeros((1, 1, *GRID_SIZE), device=DEVICE)
spins = torch.randint(0, 2, (1, 1, *GRID_SIZE), device=DEVICE, dtype=torch.float32)

# 2. Define 3D 6-neighbor Laplacian Kernel
# Neighbor weights = +1, Center weight = -6
laplacian_kernel = (
    torch.tensor(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)  # Shape: (1, 1, 3, 3, 3)

start_time = time.time()

# 3. Vectorized GPU Simulation Loop
for step in range(STEPS):
    # Vectorized 3D Laplacian via 3D Convolution
    laplacian = F.conv3d(voltage, laplacian_kernel, padding=1)

    # Reaction-Diffusion Update
    voltage = voltage + DIFFUSION_RATE * laplacian - DECAY_RATE * voltage

    # Random Stimulus
    stim_mask = (torch.rand_like(voltage) < 0.01).float()
    voltage = voltage + stim_mask * STIMULUS_STRENGTH

    # Clamp Voltages between 0 and 1
    voltage = torch.clamp(voltage, 0.0, 1.0)

    # Bistable Cell Differentiation
    states = torch.where(voltage > THRESHOLD, torch.ones_like(states), states)
    states = torch.where(voltage < (THRESHOLD / 2.0), torch.zeros_like(states), states)

    # Neighbor Spin Averaging via 3D Convolution Kernel
    spin_kernel = (
        torch.tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
            device=DEVICE,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        / 6.0
    )

    neighbor_avg_spin = F.conv3d(spins, spin_kernel, padding=1)

    # Quantum Tunneling Probability Calculation
    coupling_effect = COUPLING_STRENGTH * (1.0 - torch.abs(spins - neighbor_avg_spin))
    energy_barrier = torch.abs(0.5 - voltage) - coupling_effect
    tunneling_prob = torch.exp(-torch.clamp(energy_barrier, min=0.0) / KT)

    # Parallel Spin Flipping
    flip_mask = torch.rand_like(spins) < tunneling_prob
    spins = torch.where(flip_mask, 1.0 - spins, spins)

torch.cuda.synchronize()
elapsed = time.time() - start_time

print(f"Completed {STEPS} steps on 3D grid in {elapsed:.4f} seconds!")
print(f"Differentiated Cell Count: {int(torch.sum(states).item())}")