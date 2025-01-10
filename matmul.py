from numba import cuda
import numpy as np
import time

# Define the CUDA kernel for matrix multiplication
@cuda.jit
def matmul(A, B, C):
    # Get the thread indices
    row, col = cuda.grid(2)
    
    # Check if the thread is within bounds
    if row < C.shape[0] and col < C.shape[1]:
        # Compute the dot product
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# Size of the matrices (e.g., 4096x4096)
N = 4096

# Generate random matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Allocate memory on the device (GPU)
A_device = cuda.to_device(A)
B_device = cuda.to_device(B)
C_device = cuda.device_array((N, N), dtype=np.float32)

# Define grid and block dimensions
threads_per_block = (16, 16)
blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Start timing
start = time.time()

# Launch the kernel
matmul[blocks_per_grid, threads_per_block](A_device, B_device, C_device)

# Copy the result back to the host (CPU)
C_device.copy_to_host(C)

# Stop timing
end = time.time()

print(f"Matrix multiplication completed in {end - start:.2f} seconds")

# Verify the result (optional, for correctness)
# Uncomment the following lines to verify the result, but note that this will be slow for large matrices
#C_expected = np.dot(A, B)
#assert np.allclose(C, C_expected)
#print("Verification passed!")
