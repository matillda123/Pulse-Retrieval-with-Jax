# Pulse-Retrieval with Jax

Uses Jax to implement various algorithms for pulse-retrieval in ultrafast optics.  
Available methods are FROG, Time-Domain-Ptychography, 2D-SI, VAMPIRE and Chirp-Scans based in material dispersion or pulse-shaping.  
For each method SHG, THG, PG and SD as well as cross-correlation and interferometric are implemented (with some exceptions).

The available algorithms are: 


| Algorithm | Citation |
|---|---|
| Generalized Projection | K. W. DeLong et al., Opt. Lett. 19, 2152-2154 (1994)  |
| Ptychographic Iterative Engine | A. Maiden et al., Optica 4, 736-745 (2017) |
| Common Pulse Retrieval Algorithm | N. C. Geib, Optica 6, 495-505 (2019) |
| Differential Evolution | J. Qiang and C. Mitchell (2014).  IEEE Transactions on Evolutionary Computation. |
| Linesearch-Frog-Algorithm (for all methods) | C. O. Krook and V. Pasiskevicius, Opt. Express 33, 33258-33269 (2025)  |
| AD-Solvers (optax and optimistix) | [optax](https://github.com/google-deepmind/optax), [optimistix](https://github.com/patrick-kidger/optimistix) |
| Evolutionary solvers (evosax) | [evosax](https://github.com/RobertTLange/evosax) |


For some methods specific additional algorithms are implemented:

| Method | Algorithms | Citation |
|---|---|---|
| FROG | Vanilla | R. Trebino, 10.1007/978-1-4615-1181-6 (2000) |
|  | LSGPA | J. Gagnon et al., Appl. Phys. B 92, 25-32, 10.1007/s00340-008-3063-x (2008) |
|  |  CPCGPA | D. J. Kane and A. B. Vakhtin, Prog. Quantum Electron. 81 (100364),  (2022) |
|  |  |  |
| 2D-SI | Direct-Reconstruction | J. R. Birge et al., Opt. Lett. 31, 2063-2065, 10.1364/OL.31.002063 (2006) |
