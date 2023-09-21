import pandas as pd 
path="/root/data/siim_png_convert/k_fold.csv"
import torch
# print (len(df[df["has_pneumo"]==1]), len(df[df["has_pneumo"]!=1]),len(df[df["has_pneumo"]==1])/len(df[df["has_pneumo"]!=1]) )

y_true = torch.rand(3, 4)  # Replace with your actual tensor

# Sum the tensor along dimension 1 (rows)
sum_result = y_true.sum(dim=1)

# Check if the sum is greater than 0 for each row
is_greater_than_zero = (sum_result > 0).float()

print (is_greater_than_zero)