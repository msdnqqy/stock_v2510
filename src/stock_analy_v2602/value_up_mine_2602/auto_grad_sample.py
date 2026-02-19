import torch 
x = torch.tensor(2.0,requires_grad= True)
y = x**2 +5

y.backward()

print(x.grad)