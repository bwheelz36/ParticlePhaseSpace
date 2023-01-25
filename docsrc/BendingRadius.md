# Express bending radius in kinetic energy

The bending radius of a particle in a magnetic field is given by:

$$
\begin{equation}
\label{eqn: equation 1}
\tag{1}
r = \frac{p}{qB} = \frac{\gamma m_0 v}{qB} = \frac{\gamma \beta m_0 c}{qB}
\end{equation}
$$

Meanwhile, relatavistic kinetic energy is given by:

$$
E_k = mc^2 - m_0c^2 = \gamma m_0 c^2 - m_0 c^2 = m_0 c^2 (\gamma - 1) = m_0 c^2 (\frac{1}{\sqrt{1-\beta^2}} -1)
$$

Our goal is to express the bending radius in terms of kinetic energy.

Firstly, we can use the [momentum-energy relation](https://en.wikipedia.org/wiki/Energy%E2%80%93momentum_relation) to express momentum in terms of kinetic energy:

$$
p^2 = E^2 - m_0^2c^4
\\E = Ek + E0
\\E0 = m_0c^2
\\::p^2=(E_k+E_0)^2 - E_0^2
\\=Ek^2+2.E_k.E_0+E_0^2 - E_0^2
\\=Ek^2+2E_k.E_0
\\p=\sqrt{Ek^2+2E_k.E_0}
$$
Now, assuming we express energy in terms of eV, the momentum we have just calculated will be in units of eV/c. to use this directly in equation 1, we would have to convert to the SI units of kg.m/s:
$$
p_{SI} = p_{eV} * q/c
\\p_{SI} = \frac{q}{c}\sqrt{Ek^2+2E_k.E_0}
$$
Substituting into equation 1:
$$
r = \frac{\frac{q}{c}\sqrt{Ek^2+2E_k.E_0}}{qB}
\\=\frac{1}{Bc}\sqrt{Ek^2+2E_k.E_0}
$$

> **NOTE** that this formula assumes the energy is expressed in terms of electron volts

We can rearrange this equation slightly:
$$
r = \frac{1}{Bc}\sqrt{Ek^2+2E_k.E_0} = \frac{1}{Bc}\sqrt{Ek^2[1+2.E_0/E_k]}
\\=\frac{E_k}{Bc}\sqrt{1+2.E_0/E_k}
$$
At this point we are still assuming that energy is defined in eV. If we instead assume energy is in Joules:
$$
\\=\frac{E_k}{Bcq}\sqrt{1+2.E_0/E_k}
$$
Which is the same as Magdalena's answer!

Let's just check that this all makes sense with some basic calculations

```python
import numpy as np
from scipy import constants

q = constants.elementary_charge # C
c = constants.c  # m/s
B = 2  # T
Ek = 10e6 # eV
Ek_si = Ek*q  # J
E0 = 0.511e6  # eV
E0_si = E0*q  # J

# first check the version using eV:
r_brendan = (1/(B*c)) * np.sqrt(Ek**2+(2*Ek*E0))

# now the version using Joules:
r_magdalena = (Ek_si/(B*c*q)) * np.sqrt(1+(2*E0_si/Ek_si))

print(f'\n\nbending radius for a particle of'
      f'\nenergy: {Ek/1e6: 1.2f} MeV'
      f'\ncharge: {q: 1.2e}'
      f'\nIn a field of {B :1.2f} T'
      f'\nis (brendan:) {r_brendan:1.3f} m'
      f'\n(magdalena:) {r_magdalena: 1.3f} m')
```

```
bending radius for a particle of
energy:  10.00 MeV
charge:  1.60e-19
In a field of 2.00 T
is (brendan:) 0.018 m
(magdalena:)  0.018 m
```

The only other thing worth noting here is that for a light particle such as electrons, the ratio of rest energy to kinetic energy becomes small at high energies. At 10 MeV, the the second term in the sqrt only has a value of 0.1. So just ignoring this term we can write the bending radius much more simply:  

$$
r\approx\frac{E_k}{Bc}
$$

```python
r_approximate = Ek/(B*c)
print(f'approximate bending radius: {r_approximate: 1.3f} m'
     f'\nwhich is wrong by {100 - 100*abs(r_approximate)/r_magdalena: 1.2f} %')
```

```
approximate bending radius:  0.017 m
which is wrong by  4.75 %
```

