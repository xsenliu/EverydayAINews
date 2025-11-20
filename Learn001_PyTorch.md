[Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

```py
import torch
import numpy as np

#init
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# np array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# like
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones like Tensor: \n {x_ones} \n")
# like and override
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random like Tensor: \n {x_rand} \n")

# shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print("shape:")
print(f"Random Tensor: \n {rand_tensor} ")
print(f"Ones Tensor: \n {ones_tensor} ")
print(f"Zeros Tensor: \n {zeros_tensor}")
```
