import numpy as np
import matplotlib.pyplot as plt

class QubitStateTomography():
    """The Qubit State Tomography class. The class is used to obtain
    the state in terms of the basis state elements.
    
    The basis set considered for the states is B_rho = {I, X, Y, Z}.
    
    If the given state is rho, then rho can be written as
    
                rho = a0*I + a1*X + a2*Y + a3*Z
    
    The run() method of the class returns the list [a0, a1, a2, a3] 
    of coefficients of the basis state elements for the given state.
    """
    
    def __init__(self, state):
        """Initialize the qubit state tomography experiment."""
        self.state = state
        self.basis = self._get_paulis()
        self.exp_vals = None
    
    @staticmethod
    def _get_paulis():
        """Returns a list of Pauli matrices."""
        Hx = (1/2)*np.array([[0,1],[1,0]])
        Hy = (1/2)*np.array([[0,-1j],[1j,0]])
        Hz = (1/2)*np.array([[1,0],[0,-1]])
        Hi = (1/2)*np.array([[1,0],[0,1]])

        return [Hi, Hx, Hy, Hz]

    @staticmethod
    def _get_pauli_by_index(idx):
        """Returns the Pauli matrix corresponding to the index 'idx'."""
        Hx = (1/2)*np.array([[0,1],[1,0]])
        Hy = (1/2)*np.array([[0,-1j],[1j,0]])
        Hz = (1/2)*np.array([[1,0],[0,-1]])
        Hi = (1/2)*np.array([[1,0],[0,1]])
        p_list = [Hi, Hx, Hy, Hz]

        return p_list[idx]
    
    @staticmethod
    def _expectation(op, state):
        """Returns the expectation <state|op|state>."""
        conj_state = np.conj(state)
        return conj_state.dot(np.matmul(op, state))

    def run(self):
        """Returns the list of coefficients corresponding to the decomposition."""
        exp_vals = []

        for op in self.basis:
            coeff = self._expectation(op, self.state)
            exp_vals.append(coeff)
        self.exp_vals = np.array(exp_vals)

        return exp_vals