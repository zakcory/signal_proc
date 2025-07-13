import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, svd


def create_synthetic_vector(size, intensity):
    base = np.random.randn() * np.sqrt(intensity)
    spike = np.random.randn() * np.sqrt(size * (1 - intensity) / 2)
    pos = np.random.randint(0, size // 2)

    vector = np.full(size, base)
    vector[pos] += spike
    vector[pos + size // 2] += spike
    return vector


def estimate_cov(samples):
    n_samples, n_features = samples.shape
    cov_matrix = sum(np.outer(sample, sample) for sample in samples)
    return cov_matrix / n_samples



def add_awgn(data, sigma2):
    return data + np.random.randn(*data.shape) * np.sqrt(sigma2)


def build_degradation_matrix(size):
    kernel = np.zeros(size)
    kernel[0:3] = [-5 / 2, 4 / 3, -1 / 12]
    kernel[-2:] = [-1 / 12, 4 / 3]
    return toeplitz(kernel, kernel)


def compute_wiener_filter(true_data, noisy_data, R_signal, sigma2, H):
    R_noise = sigma2 * np.eye(R_signal.shape[0])
    denominator = H @ R_signal @ H.T + R_noise
    W = R_signal @ H.T @ np.linalg.pinv(denominator)
    estimate = noisy_data @ W.T

    mse_element = np.mean((true_data - estimate) ** 2)
    mse_vector = np.mean(np.sum((true_data - estimate) ** 2, axis=1))

    return R_noise, W, estimate, mse_element, mse_vector


def apply_wiener_filter(true_data, noisy_data, R_signal, sigma2, H):
    R_noise, W, estimate, mse_element, mse_vector = compute_wiener_filter(
        true_data, noisy_data, R_signal, sigma2, H
    )

    print(f"--- Wiener Filtering (σ² = {sigma2:.2f}) ---")
    print(f"Per-element MSE: {mse_element:.4f}")
    print(f"Per-signal MSE : {mse_vector:.4f}\n")

    return R_noise, W, estimate

def display_signal_comparison(original, noisy, restored, tag="Signal"):
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ids = np.random.choice(original.shape[0], 5, replace=False)

    for i, idx in enumerate(ids):
        axes[i].plot(original[idx], label="Original", color="darkgreen", lw=1.5)
        axes[i].plot(noisy[idx], label="Noisy", color="crimson", ls="--", lw=1)
        axes[i].plot(restored[idx], label="Restored", color="navy", ls="-.", lw=1.2)
        axes[i].set_title(f"{tag} #{i + 1}")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, ls=":", alpha=0.5)

    plt.xlabel("Index")
    plt.tight_layout()
    plt.show()


def plot_matrix(mat, title, label="Value"):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, cmap='plasma', aspect='auto')
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(label, rotation=270, labelpad=15)
    plt.xlabel("Col")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()



def step_a_generate_ensemble(N, c, count):
    print("\n--- Step A: Generating Signals ---")
    dataset = np.array([create_synthetic_vector(N, c) for _ in range(count)])
    mean_vec = np.mean(dataset, axis=0)
    cov_mat = estimate_cov(dataset)

    plt.figure(figsize=(10, 4))
    plt.plot(mean_vec, label="Mean Vector")
    plt.grid(True, ls=":")
    plt.title("Mean Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_matrix(cov_mat, "Corr", "Cor")
    return dataset, cov_mat


def step_b_basic_denoising(data, R_signal, sigma2, N):
    noisy = add_awgn(data, sigma2)
    _, W, estimate = apply_wiener_filter(data, noisy, R_signal, sigma2, np.eye(N))
    display_signal_comparison(data, noisy, estimate, tag=f"B")
    plot_matrix(W, f"Wiener Matrix (σ²={sigma2})", "Weight")


def step_c_degraded_denoising(data, R_signal, sigma2, N):
    H = build_degradation_matrix(N)
    degraded = np.einsum('ij,nj->ni', H, data)
    noisy = add_awgn(degraded, sigma2)
    _, W, estimate = apply_wiener_filter(data, noisy, R_signal, sigma2, H)
    display_signal_comparison(data, noisy, estimate, tag=f"C")
    plot_matrix(W, f"Wiener Matrix (Degraded σ²={sigma2})", "Weight")
    return H




def run_pipeline():
    N = 64
    c = 0.6
    count = 5000

    data, R = step_a_generate_ensemble(N, c, count)

    step_b_basic_denoising(data, R, sigma2=1.0, N=N)
    step_c_degraded_denoising(data, R, sigma2=1.0, N=N)

    step_b_basic_denoising(data, R, sigma2=5.0, N=N)
    step_c_degraded_denoising(data, R, sigma2=5.0, N=N)


if __name__ == '__main__':
    run_pipeline()
