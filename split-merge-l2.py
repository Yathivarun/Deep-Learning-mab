import torch

# --------------------------------------------------------
# 1. CREATE A BASIC TENSOR
# --------------------------------------------------------
x = torch.tensor([10, 20, 30, 40, 50, 60])
# Expected: tensor([10, 20, 30, 40, 50, 60])


# --------------------------------------------------------
# 2. SPLIT OPERATIONS
# --------------------------------------------------------

# Split tensor into 3 equal parts
s1 = torch.split(x, 2)  
# Expected: (tensor([10,20]), tensor([30,40]), tensor([50,60]))

# Split by indices
s2 = torch.tensor_split(x, [2, 4])  
# Expected: [tensor([10,20]), tensor([30,40]), tensor([50,60])]


# --------------------------------------------------------
# 3. MERGE OPERATIONS (Concatenate / Stack)
# --------------------------------------------------------

# Concatenate → joins tensors along existing dimension
c1 = torch.cat([s1[0], s1[1], s1[2]])
# Expected: tensor([10, 20, 30, 40, 50, 60])

# Stack → creates new dimension
c2 = torch.stack([s1[0], s1[1], s1[2]])
# Expected:
# tensor([[10,20],
#         [30,40],
#         [50,60]])


# --------------------------------------------------------
# 4. STATISTICAL OPERATIONS
# --------------------------------------------------------

mean_val = x.mean()        # Expected: tensor(35.)
sum_val = x.sum()          # Expected: tensor(210)
min_val = x.min()          # Expected: tensor(10)
max_val = x.max()          # Expected: tensor(60)
std_val = x.std()          # Standard deviation
argmax_val = x.argmax()    # Expected: tensor(5) → index of max value
argmin_val = x.argmin()    # Expected: tensor(0)


# --------------------------------------------------------
# PRINT ALL RESULTS
# --------------------------------------------------------
print("Original Tensor:", x)
print("\nSplit (equal parts):", s1)
print("Split (index based):", s2)

print("\nConcatenate:", c1)
print("Stack:\n", c2)

print("\nStatistics:")
print("Mean:", mean_val)
print("Sum:", sum_val)
print("Min:", min_val)
print("Max:", max_val)
print("Std:", std_val)
print("Argmax:", argmax_val)
print("Argmin:", argmin_val)





"""
Original Tensor: tensor([10, 20, 30, 40, 50, 60])

Split (equal parts): 
(tensor([10, 20]), tensor([30, 40]), tensor([50, 60]))

Split (index based): 
[tensor([10, 20]), tensor([30, 40]), tensor([50, 60])]

Concatenate: tensor([10, 20, 30, 40, 50, 60])

Stack:
 tensor([[10, 20],
        [30, 40],
        [50, 60]])

Statistics:
Mean: tensor(35.)
Sum: tensor(210)
Min: tensor(10)
Max: tensor(60)
Std: tensor(17.078...)
Argmax: tensor(5)
Argmin: tensor(0)

"""
