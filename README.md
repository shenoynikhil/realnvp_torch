## Pytorch Implementation of RealNVP
Basic pytorch implementation of RealNVP for higher dimensional images based on the paper <a href = "https://arxiv.org/abs/1605.08803">Density estimation using Real NVP</a>.

The pseudocode for the realNVP class ,
```
Preprocess() # dequantization, logit trick from RealNVP (Dinh et al) Section 4.1 (mentioned in utils.py)

for _ in range(4):
  AffineCouplingWithCheckerboard() # Figure 3 in Dinh et al - (left)
  ActNorm() # described in Glow (Kingma & Dhariwal) Section 3.1
Squeeze(), # [b, c, h, w] --> [b, c*4, h//2, w//2]

for _ in range(3):
  AffineCouplingWithChannel()
  ActNorm()
Unsqueeze(), # [b, c*4, h//2, w//2] --> [b, c, h, w]

for _ in range(3):
  AffineCouplingWithCheckerboard()
  ActNorm()
```

The pseudocode for the coupling layers is, 
```
ResnetBlock: n_filters
  h = x
  h = conv2d(n_filters, n_filters, (1,1), stride=1, padding=0)(h)
  h = relu(h)
  h = conv2d(n_filters, n_filters, (3,3), stride=1, padding=1)(h)
  h = relu(h)
  h = conv2d(n_filters, n_filters, (1,1), stride=1, padding=0)(h)
  return h + x

SimpleResnet: n_filters = 256, n_blocks = 8, n_out
  conv2d(in_channels, n_filters=128, (3,3), stride=1, padding=1)
  apply 8 ResnetBlocks with n_filters=128
  relu()
  conv2d(n_filters, n_filters=n_out, (3,3), stride=1, padding=1)

AffineCoupling(x, mask):
  x_ = x * mask
  log_s, t = torch.chunk(SimpleResnet(x_), 2, dim=1)
  t = t * (1.0 - mask)
  log_scale = log_scale * (1.0 - mask)
  z = x * torch.exp(log_scale) + t
  log_det_jacobian = log_scale
  return z, log_det_jacobian
```

#### Results
The dataset used is CIFAR10 and the samples were generated after 5, 10, 15 and 20 epochs. The images generated for CIFAR10 are as follows,

#### References
1. https://github.com/fmu2/realNVP
2. Implementation of Real_NVP in pytorchhttps://github.com/chrischute/real-nvp
3. CS294 Deep Unsupervised Learning Course - Assignment 2 https://sites.google.com/view/berkeley-cs294-158-sp20/home

