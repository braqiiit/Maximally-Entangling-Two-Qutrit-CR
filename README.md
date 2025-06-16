# Realization of Maximally Entangling Two Qutrit Gates Using the Cross-Resonance Scheme

Superconducting transmons, often truncated to two levels for qubit operations, have an inherent multi-level energy structure that offers untapped potential for higher-dimensional quantum systems like qutrits.
Three-level systems have natural advantages over their two-level counterparts in quantum information and computation.
However, universal qutrit computation requires high-fidelity entangling gates spanning all three levels, which remain experimentally challenging. 
In this letter, we introduce the generalized cross-resonance scheme (GCR) which is a comprehensive theoretical framework that generalize the qubit-centric cross-resonance (CR) interaction beyond the 0-1 subspace for realizing maximally entangling two-qutrit gates on fixed-frequency transmons and is a microwave-only technique compatible with existing hardware. 
We use the GCR scheme to design parametric two-qutrit gates, namely, $U_{CR}^{01}$ and $U_{CR}^{12}$, that act on the $0{-}1$ and $1{-}2$ energy transitions of transmons.
Our gates improve upon the existing works in two aspects.
First, our gates *directly* allow for entanglement on the $1{-}2$ levels rather than merely relying on $0{-}1$ entanglement, as in previous works. Second, our gates are parametric in nature, enabling us to construct multiple entangling gates of interest, whereas the purview of prior works that use cross-resonance for qutrits was limited to individual gates. 
Using numerical simulation in Qiskit Dynamics, we demonstrate two-qutrit generalized controlled $X$ ($U_{CX}^{01}$ and $U_{CX}^{12}$) and controlled $H$ ($U_{CH}^{01}$ and $U_{CH}^{12}$) gates, which are instances of the proposed $U_{CR}$ gates, with reported gate fidelities of $99.73\pm 0.01\%, 97.88\pm 0.01\%, 99.39\pm 0.01\%$, and $98.99\pm 0.01\%$, respectively.
Finally, we prepare a two-qutrit Bell state $|\psi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)$ with a fidelity of $99.06 \pm 0.01\%$. 
We note that, in our setup, the complete time taken for Bell state preparation is $\sim 514$ ns and is less than the gate time of cross-Kerr-based entangling gates.  

This repository contains the code corresponding to the experiments that were performed as part of the paper. All the implementations are present in the Jupyter notebooks. The notebooks are self-explanatory.
In case of any queries, please feel free to contact Tharrma at tharrmashasthav@iiitd.ac.in.
