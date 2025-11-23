import torch

# -----------------------------
# 1. CREATE BASIC TENSORS
# -----------------------------

# Tensor from a Python list
a = torch.tensor([1, 2, 3])
# Expected output: tensor([1, 2, 3])

# Tensor of zeros (shape: 2x3)
b = torch.zeros((2, 3))
# Expected output: 
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# Tensor of ones (shape: 2x2)
c = torch.ones((2, 2))
# Expected output:
# tensor([[1., 1.],
#         [1., 1.]])

# Random tensor (values between 0 and 1)
d = torch.rand((2, 2))
# Expected output: random 2x2 values


# -----------------------------
# 2. BASIC OPERATIONS
# -----------------------------

# Element-wise addition
add_result = a + torch.tensor([10, 10, 10])
# Expected: tensor([11, 12, 13])

# Element-wise multiplication
mul_result = a * 2
# Expected: tensor([2, 4, 6])

# Matrix multiplication
m1 = torch.tensor([[1, 2],
                   [3, 4]])
m2 = torch.tensor([[5, 6],
                   [7, 8]])

matmul_result = m1 @ m2
# Expected:
# tensor([[19, 22],
#         [43, 50]])

# -----------------------------
# 3. TENSOR PROPERTIES
# -----------------------------

shape_val = a.shape        # Expected: torch.Size([3])
dtype_val = a.dtype        # Expected: torch.int64
device_val = a.device      # Expected: cpu (unless using GPU)

# -----------------------------
# 4. RESHAPING OPERATIONS
# -----------------------------

tensor_x = torch.arange(1, 7)  # tensor([1, 2, 3, 4, 5, 6])

reshaped = tensor_x.reshape(2, 3)
# Expected:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

flattened = reshaped.flatten()
# Expected: tensor([1, 2, 3, 4, 5, 6])

# -----------------------------
# PRINT ALL RESULTS
# -----------------------------
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("Addition:", add_result)
print("Multiplication:", mul_result)
print("Matrix Multiplication:\n", matmul_result)
print("Shape:", shape_val)
print("Dtype:", dtype_val)
print("Device:", device_val)
print("Reshaped:\n", reshaped)
print("Flattened:", flattened)



"""
a = tensor([1, 2, 3])
b = tensor([[0., 0., 0.],
            [0., 0., 0.]])
c = tensor([[1., 1.],
            [1., 1.]])
d = tensor([[0.4219, 0.8761],
            [0.2919, 0.7120]])

Addition: tensor([11, 12, 13])
Multiplication: tensor([2, 4, 6])

Matrix Multiplication:
 tensor([[19, 22],
        [43, 50]])

Shape: torch.Size([3])
Dtype: torch.int64
Device: cpu

Reshaped:
 tensor([[1, 2, 3],
        [4, 5, 6]])

Flattened: tensor([1, 2, 3, 4, 5, 6])

"""
