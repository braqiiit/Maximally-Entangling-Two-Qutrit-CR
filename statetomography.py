import numpy as np
import matplotlib.pyplot as plt

from helper_pulse_functions import get_i_pulse_0 as get_i_pulse, get_meas_basis_pulse_0 as get_meas_basis_pulse
from helper_pulse_functions import get_i_pulse_1, get_meas_basis_pulse_1

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
        """Initialize the qubit state tomography experiment.
        
        Args:
            state: The state whose tomography is to be performed.
              The state can be provided as a vector or a density
              matrix.
        """
        if np.array(state).shape == (2,):
            self.state = np.outer(state, state)
        else:
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
        return np.trace(np.matmul( state.conj().T, np.matmul(op, state)))

    def run(self):
        """Returns the list of coefficients corresponding to the decomposition."""
        exp_vals = []

        for op in self.basis:
            coeff = self._expectation(op, self.state)
            exp_vals.append(coeff)
        self.exp_vals = np.array(exp_vals)

        return exp_vals
    

class TwoQutritStateTomography():
    """The Two-Qutrit State Tomography class. The class is used to obtain
    the state in terms of the basis state elements.
    
    The basis set considered for the states is B_rho = {sigma_i X sigma_j : i,j in [8]}.
    
    If the given state is rho, then rho can be written as
    
                rho = sum_{i,j} a_ij*(sigma_i X sigma_j)
    
    The run() method of the class returns the list [a_00, a_01, ..., a_88] 
    of coefficients of the basis state elements for the given state.
    """
    
    def __init__(self, state, backend=None):
        """Initialize the two qutrit state tomography experiment.
        
        Args:
            state: The state whose tomography is to be performed.
              The state should be provided as a vector.
        """
        if np.array(state).shape != (9,):
            raise Exception("The given state is not a vector corresponding to a valid two-qutrit state.")
        self.state = state
        self.backend = backend
        self.qst_matrix = None

    @staticmethod
    def _get_basis_0():
        basis_set = []
        subspace = ['01', '02', '12']
        # For I operator.
        basis_set.append(get_i_pulse_0())
        for sub in subspace:
            for sdg in [False, True]:
                basis_set.append(get_meas_basis_pulse(sub, sdg, measure=True))
            if sub=='01':
                # For lambda_3.
                basis_set.append(get_i_pulse())
        # For lambda_8.
        basis_set.append(get_i_pulse())

        return basis_set

    @staticmethod
    def _get_basis_1():
        basis_set = []
        subspace = ['01', '02', '12']
        # For I operator.
        basis_set.append(get_i_pulse_1())
        for sub in subspace:
            for sdg in [False, True]:
                basis_set.append(get_meas_basis_pulse_1(sub, sdg, measure=True))
            if sub=='01':
                # For lambda_3.
                basis_set.append(get_i_pulse_1())
        # For lambda_8.
        basis_set.append(get_i_pulse_1())

        return basis_set
    
    @staticmethod
    def _get_meas_basis_matrix():
        c = 1/np.sqrt(2)

        std_basis  = np.array([[1,0,0],[0,1,0],[0,0,1]])

        x_basis_01 = np.array([[c,c,0],[c,-c,0],[0,0,1]])
        y_basis_01 = np.array([[c,-1j*c,0],[c,1j*c,0],[0,0,1]])

        x_basis_02 = np.array([[c,0,c],[0,1,0],[c,0,-c]])
        y_basis_02 = np.array([[c,0,-1j*c],[0,1,0],[c,0,1j*c]])

        x_basis_12 = np.array([[1,0,0],[0,c,c],[0,c,-c]])
        y_basis_12 = np.array([[1,0,0],[0,c,-1j*c],[0,c,1j*c]])

        mats = [std_basis, x_basis_01, y_basis_01, std_basis, x_basis_02, y_basis_02, x_basis_12, y_basis_12, std_basis]

        return mats

    @staticmethod
    def _get_gm_coeff_lambda(x):
        c = np.array(np.sqrt(3/2), dtype='complex')
        d = np.array(1/3, dtype='complex')
        c_list = [d, c*d, c*d, c*d, c*d, c*d, c*d, c*d, c*d]

        return c_list[x]
    
    @staticmethod
    def _get_probs(state):
        return [np.absolute(vec)**2 for vec in state]
    
    @staticmethod
    def _get_eigenvals():
        evals_0 = [(1,1,1), (1,-1,0), (1,-1,0), (1,-1,0), (1,0,-1), (1,0,-1), (0,1,-1), (0,1,-1), (1/np.sqrt(3), 1/np.sqrt(3), -2*np.sqrt(1/3))]
        evals_1 = [(1,1,1), (1,-1,0), (1,-1,0), (1,-1,0), (1,0,-1), (1,0,-1), (0,1,-1), (0,1,-1), (1/np.sqrt(3), 1/np.sqrt(3), -2*np.sqrt(1/3))]

        evals = []

        for ev0 in evals_0:
            for ev1 in evals_1:
                evals.append(np.kron(np.array(ev1, dtype=complex), np.array(ev0, dtype=complex)))

        return evals
    
    def run(self, shots=1000, mode='solve', full_pulse=False):
        """ The function that returns the QST coefficients corresponding to the given state."""
        
        if mode not in ['run', 'solve']:
            raise Exception(f"Provide mode {mode} is not one of 'run' or 'solve'")

        probs = []
        basis_vecs = ['00', '01', '02', '10', '11', '12', '20', '21', '22']

        if full_pulse:
            if self.backend is None:
                raise Exception("Please provide a backend when full_pulse is True.")
            basis_set_0 = self._get_basis_0()
            basis_set_1 = self._get_basis_1()

            for i, basis0 in enumerate(basis_set_0):
                print(i)
                for j, basis1 in enumerate(basis_set_1):
                    basis = basis0.append(basis1, inplace=False)
                    if mode=='run':
                        job = self.backend.run(basis, shots=shots)
                        cnt = job.result().get_counts()
                    elif mode=='solve':
                        job = self.backend.solve(basis, y0=self.state, t_span=t_span)
                        res_state = job[0].y[-1]
                        cnt = get_probs_dict(res_state)
                    prob_arr = []
                    for vec in basis_vecs:
                        if vec in cnt:
                            if mode=='run':
                                vec_prob = cnt[vec]/shots
                            else:
                                vec_prob = cnt[vec]
                            prob_arr.append(vec_prob)
                        else:
                            prob_arr.append(0)
                    probs.append(prob_arr)
        else:
            res_state = self.state
            basis_matrices = self._get_meas_basis_matrix()
            basis_meas_outcomes = {}
            for mat0 in basis_matrices:
                for mat1 in basis_matrices:
                    meas_mat = np.kron(mat1, mat0)
                    fin_state = np.matmul(meas_mat, res_state)
                    prob_arr = self._get_probs(fin_state)
                    probs.append(prob_arr)        

        eigenvals = self._get_eigenvals()

        coeffs = []
        for j, (prob, ev) in enumerate(zip(probs, eigenvals)):
            coeff = 0
            for i in range(9):
                coeff += prob[i]*ev[i]*self._get_gm_coeff_lambda(j//9)*self._get_gm_coeff_lambda(j%9)
            coeffs.append(coeff)

        qst_matrix = []
        for i in range(9):
            qst_matrix.append([])
            for j in range(9):
                qst_matrix[i].append(coeffs[(9*i)+j])
        
        self.qst_matrix = np.array(qst_matrix)
        
        return self.qst_matrix
    
    def qst_matrix_to_density_matrix(self, qst_matrix=None):
        if qst_matrix is None and self.qst_matrix is None:
            raise Exception("Use .run to obtain the qst_matrix before converting to density matrix.")
        
        if qst_matrix is None and self.qst_matrix is not None:
            qst_matrix = self.qst_matrix
        
        c = np.array(np.sqrt(3/2), dtype='complex')
        d = np.array(1/3, dtype='complex')

        lam = [
            d*np.array([[1,0,0],[0,1,0],[0,0,1]]),
            d*c*np.array([[0,1,0],[1,0,0],[0,0,0]]), 
            d*c*np.array([[0,-1j,0],[1j,0,0],[0,0,0]]),
            d*c*np.array([[1,0,0],[0,-1,0],[0,0,0]]),
            d*c*np.array([[0,0,1],[0,0,0],[1,0,0]]),
            d*c*np.array([[0,0,-1j],[0,0,0],[1j,0,0]]),
            d*c*np.array([[0,0,0],[0,0,1],[0,1,0]]),
            d*c*np.array([[0,0,0],[0,0,-1j],[0,1j,0]]),
            d*c*(1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]])
        ]

        rho = None
        for i, mat0 in enumerate(lam):
            for j, mat1 in enumerate(lam):
                if rho is None:
                    rho = qst_matrix[i][j]*np.kron(mat1, mat0)
                else:
                    rho = rho + qst_matrix[i][j]*np.kron(mat1, mat0)        
        return rho