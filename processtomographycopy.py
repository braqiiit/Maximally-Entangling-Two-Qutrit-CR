import numpy as np
import sympy
from sympy import KroneckerDelta

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from helper_pulse_functions import get_i_pulse, get_meas_basis_pulse


class QubitProcessTomography():
    """ The Qubit Process Tomography class. The class is used to obtain
    the process matrix for any given single qubit gate.
    
    The basis set considered for the process is B_E = {I, X, Y, Z}.
    The basis set considered for the states is B_rho = {I, X, Y, Z}.
    
    The process matrix can be obtained by executing the .run() method of
    the class object.
    """
    def __init__(self, gate=None):
        """Initialize a Quantum Process Tomography experiment.
        Args:
            gate: The matrix representation of the gate whose
              tomography is to be performed. If no gate is provided
              an Identity gate is considered.
        """
        self.gate = gate
        self.lambda_arr = None
        self.chi_matrix = None
        self.beta = None
        self.beta_matrix = self.construct_beta_matrix()
        
    l0, l1, l2, l3 = sympy.symbols('l0 l1 l2 l3', commutative=False)
    l_vec = [l0, l1, l2, l3]
    
    def construct_beta_matrix(self):
        """ The function that constructs the 16x16 beta coefficient matrix
        from the beta values.
        """
        if self.beta is None:
            self.beta = self.get_beta_values()

        beta_matrix = []
        for j in range(4):
            for k in range(4):
                row = []
                for m in range(4):
                    for n in range(4):
                        row.append(complex(self.beta[m][j][n][k]))
                beta_matrix.append(row)
        
        return beta_matrix
    
    
    def get_beta_values(self):
        """ The function that computes the coefficients of all basis elements
        rho_j for all possible (E_m, rho_i, E_n) tuples.
        """

        beta_1q = []

        for x in range(4):
            beta_1q.append([])
            for y in range(4):
                beta_1q[x].append([])
                for z in range(4):
                    beta_1q[x][y].append([])
                    prod_yz = sympy.expand(self.l_vec[x]*self._pauli_prod(y,z))
                    terms = sympy.Add.make_args(prod_yz)
                    new_terms = []
                    for term in terms:
                        syms = self._get_vars(term)
                        if len(syms)==1:
                            poly = syms[0]*syms[0]
                        else:
                            poly = self._get_vars_prod(syms)
                        coeff = term.coeff(poly)
                        if len(syms)==1:
                            if term.coeff(syms[0]**2)!=0:
                                prod = self._pauli_prod(int(str(syms[0])[-1]), int(str(syms[0])[-1]))
                                new_terms.append(coeff*prod)
                            else:
                                new_terms.append(term)
                        elif len(syms)==2:
                            prod = self._pauli_prod(int(str(syms[0])[-1]), int(str(syms[1])[-1]))
                            new_terms.append(coeff*prod)
                    t = 0
                    for nt in new_terms:
                        t = t+nt
                    t = sympy.expand(t)
                    for i in range(4):
                        beta_1q[x][y][z].append(t.coeff(self.l_vec[i]))

        return beta_1q     
    
    @staticmethod
    def _get_paulis():
        """Returns a list of Pauli matrices."""
        Hx = (1/2)*np.array([[0,1],[1,0]])
        Hy = (1/2)*np.array([[0,-1j],[1j,0]])
        Hz = (1/2)*np.array([[1,0],[0,-1]])
        Hi = (1/2)*np.array([[1,0],[0,1]])

        return [Hi, Hx, Hy, Hz]
    
    @staticmethod
    def _get_vars(expr):
        """Returns a list of symbols in the order that
        it occurs in 'expr'.
        """
        syms = expr.free_symbols
        syms_order = []
        for sym in syms:
            syms_order.append((sym, str(expr).index(str(sym))))
        syms_order.sort(key=lambda x: x[1])

        return [sym for (sym,_) in syms_order]

    @staticmethod
    def _get_vars_prod(syms):
        """Returns an ordered product of the symbols in syms."""
        prod = syms[0]
        for sym in syms[1:]:
            prod = prod*sym

        return prod
    
    @staticmethod
    def _eps(a,b,c):
        """The Levi-Civita symbol function."""
        if (a,b,c) in [(1,2,3), (2,3,1), (3,1,2)]:
            val = 1
        elif (a,b,c) in [(3,2,1), (1,3,2), (2,1,3)]:
            val = -1
        elif a==b or b==c or c==a:
            val = 0
        return val

    def _pauli_prod(self, a, b):
        """Returns the product of two Pauli matrices in terms of 
        Pauli Matrices."""
        if a==0:
            return self.l_vec[b]
        elif b==0:
            return self.l_vec[a]
        else:
            return KroneckerDelta(a,b)*self.l0 + 1j*self._eps(a,b,1)*self.l1 + 1j*self._eps(a,b,2)*self.l2 + 1j*self._eps(a,b,3)*self.l3
    
    @staticmethod
    def _expectation(op, state):
        """Returns the expectation <state|op|state>."""
        conj_state = np.conj(state)
        
        return conj_state.dot(np.matmul(op, state))
    
    def get_lambda(self):
        """ Returns the flattened array of the matrix obtained
        by performing a state tomography of all the basis state
        elements.
        """
        paulis = self._get_paulis()
        if self.gate is None:
            self.gate = np.eye(2)
        lambda_matrix = []
        for op in paulis:
            arr = []
            evals, evecs = np.linalg.eig(op)
            for basis in paulis:
                exp_vals = []
                for i, _ in enumerate(evecs):
                    state = np.matmul(self.gate, evecs[:,i])
                    exp_vals.append(self._expectation(basis, state))
                lambda_value = 0
                for evl, val in zip(evals, exp_vals):
                    lambda_value = lambda_value + evl*val
                arr.append(lambda_value)
            lambda_matrix.append(arr)
        
        lambda_array = []
        for i in range(4):
            for j in range(4):
                lambda_array.append(lambda_matrix[i][j])
        
        self.lambda_array = np.array(lambda_array)

        return self.lambda_array
        
    def run(self):
        """Returns the process matrix corresponding to the gate."""
        self.lambda_array = self.get_lambda()
        chi_array = spsolve(csc_matrix(self.beta_matrix), self.lambda_array)
    
        chi_matrix = []
        for i in range(4):
            chi_matrix.append([])
            for j in range(4):
                chi_matrix[i].append(chi_array[(4*i)+j])
        
        self.chi_matrix = np.array(chi_matrix)
        
        return self.chi_matrix
    
    def plot_chi(self):
        """The plotting function."""
        if self.chi_matrix is None:
            raise Exception(f"Please use .run() fuction to obtain chi matrix first.")
        
        data = self.chi_matrix
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dims = len(data[0])
        # Add spacing
        spacing = 1.5
        x_idx = np.arange(dims) * spacing
        y_idx = np.arange(dims) * spacing
        x, y = np.meshgrid(x_idx, y_idx)

        # Plot bars
        ax.bar3d(
            x.ravel(), y.ravel(), np.zeros(dims**2), 1, 1, data.ravel(),
            shade=False, alpha=0.5, edgecolor='black', linewidth=0.3
        )

        labels = ['I', 'X', 'Y', 'Z']
        
        ax.set_xticks(x_idx + 0.5)
        ax.set_yticks(y_idx + 0.5)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.tick_params(axis='z', labelsize=8)

        display(plt.show())


class QutritProcessTomography():
    """ The Qutrit Process Tomography class. The class is used to obtain
    the process matrix for any given single qutrit gate.
    
    The basis set considered for the process and for the state is the set
    of Gellmann matrices.
    
    The process matrix can be obtained by executing the .run() method of
    the class object.
    """
    def __init__(self, gate=None, gate_pulse=None, backend=None):
        """Initialize a Quantum Process Tomography experiment.
        Args:
            gate: The matrix representation of the gate whose
              tomography is to be performed. If no gate is provided
              an Identity gate is considered.
        """
        self.gate = gate
        self.gate_pulse = gate_pulse
        self.backend = backend
        self.lambda_arr = None
        self.chi_matrix = None
        self.beta = None
        self.beta_matrix = self.construct_beta_matrix()
        
    l0, l1, l2, l3, l4, l5, l6, l7, l8 = sympy.symbols('l0 l1 l2 l3 l4 l5 l6 l7 l8', commutative=False)
    l_vec = [l0, l1, l2, l3, l4, l5, l6, l7, l8]
    
#     c = np.array(3/2, dtype='complex')
#     d = np.array(1/3, dtype='complex')
    
    c = np.array(1, dtype='complex')
    d = np.array(1, dtype='complex')

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
    
    def construct_beta_matrix(self):
        """ The function that constructs the 81x81 beta coefficient matrix
        from the beta values.
        """
        if self.beta is None:
            self.beta = self.get_beta_values()

        beta_matrix = []

        for j in range(9):
            for k in range(9):
                row = []
                for m in range(9):
                    for n in range(9):
                        row.append(complex(self.beta[m][j][n][k]))
                beta_matrix.append(row)
        
        return beta_matrix
    
    
    def get_beta_values(self):
        """ The function that computes the coefficients of all basis elements
        rho_j for all possible (E_m, rho_i, E_n) tuples.
        """

        beta_vals = []

        for x in range(9):
            beta_vals.append([])
            for y in range(9):
                beta_vals[x].append([])
                for z in range(9):
                    beta_vals[x][y].append([])
                    prod_coeffs = self._get_gm_coeff(x)*self._get_gm_coeff(y)*self._get_gm_coeff(z)
                    prod_yz = sympy.expand(prod_coeffs*self.l_vec[x]*self._gm_prod(y,z))
                    terms = sympy.Add.make_args(prod_yz)
                    new_terms = []
                    for term in terms:
                        syms = self._get_vars(term)
                        if len(syms)==1:
                            poly = syms[0]*syms[0]
                        else:
                            poly = self._get_vars_prod(syms)
                        coeff = term.coeff(poly)
                        if len(syms)==1:
                            if term.coeff(syms[0]**2)!=0:
                                prod = self._gm_prod(int(str(syms[0])[-1]), int(str(syms[0])[-1]))
                                new_terms.append(coeff*prod)
                            else:
                                new_terms.append(term)
                        elif len(syms)==2:
                            prod = self._gm_prod(int(str(syms[0])[-1]), int(str(syms[1])[-1]))
                            new_terms.append(coeff*prod)

                    t = 0
                    for nt in new_terms:
                        t = t+nt
                    t = sympy.expand(t)
                    for i in range(9):
                        beta_vals[x][y][z].append(t.coeff(self.l_vec[i])/self._get_gm_coeff(i))

        self.beta = beta_vals
        
        return self.beta
    
    @staticmethod
    def _get_eigenvals():
        evals = [(1,1,1), (1,-1,0), (1,-1,0), (1,-1,0), (1,0,-1), (1,0,-1), (0,1,-1), (0,1,-1), (1/np.sqrt(3), 1/np.sqrt(3), -2*np.sqrt(1/3))]

        return evals
    
    @staticmethod
    def _get_basis():
        basis_set = []
        subspace = ['01', '12', '02']
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
    
    def _gm_prod(self,x,y):
        if x==0 and y==0:
            return self.l0
        else:
            prod = (((2/3)*KroneckerDelta(x,y)*self.l0) 
                    + (self._d_plus_f(x,y,1))*self.l1 
                    + (self._d_plus_f(x,y,2))*self.l2
                    + (self._d_plus_f(x,y,3))*self.l3 
                    + (self._d_plus_f(x,y,4))*self.l4 
                    + (self._d_plus_f(x,y,5))*self.l5 
                    + (self._d_plus_f(x,y,6))*self.l6 
                    + (self._d_plus_f(x,y,7))*self.l7 
                    + (self._d_plus_f(x,y,8))*self.l8
                   )
            
            return prod
    
    @staticmethod
    def _get_probs(state):
        return [np.absolute(vec)**2 for vec in state]
    
    @staticmethod
    def _get_probs_dict(state):
        prob_dict = {}
        for i in range(3):
            for j in range(3):
                prob_dict[str(i)+str(j)] = np.absolute(state[(3*i)+j])**2
        return prob_dict
    
    @staticmethod
    def _get_meas_basis_matrix():
        c = 1/np.sqrt(2)

        std_basis  = np.array([[1,0,0],[0,1,0],[0,0,1]])

        x_basis_01 = np.array([[c,c,0],[c,-c,0],[0,0,1]])
        y_basis_01 = np.array([[c,-1j*c,0],[c,1j*c,0],[0,0,1]])

        x_basis_12 = np.array([[1,0,0],[0,c,c],[0,c,-c]])
        y_basis_12 = np.array([[1,0,0],[0,c,-1j*c],[0,c,1j*c]])

        x_basis_02 = np.array([[c,0,c],[0,1,0],[c,0,-c]])
        y_basis_02 = np.array([[c,0,-1j*c],[0,1,0],[c,0,1j*c]])

        mats = [std_basis, x_basis_01, y_basis_01, std_basis, x_basis_02, y_basis_02, x_basis_12, y_basis_12, std_basis]

        return mats
    
    def run_experiment_for_coeffs(self, shots=1000, mode='run', init_state=None, t_span=None, gate_pulse=None, gate_matrix=None, full_pulse=False):
        if mode not in ['run', 'solve']:
            raise Exception(f"Provide mode {mode} is not one of 'run' or 'solve'")
        if mode=='solve':
            if init_state is None:
                init_state = np.array([1,0,0])
            if len(init_state)!=3:
                raise Exception(f'Initial state has dimensions {len(init_state)} not equal to 3.')
            if full_pulse:
                if t_span is None:
                    raise Exception('Time span not provided when mode=solve')
                if not isinstance(t_span, list):
                    raise Exception('Time span is not in a list.')
                elif len(t_span)!=2:
                    raise Exception(f'length of t_span is {len(t_span)} not equal to 2.')

        dt = 1/4.5e9
        probs = []
        basis_vecs = ['0', '1', '2']

        if full_pulse:
            if self.backend is None:
                raise Exception(f"Backend is not provided for the experiment.")
            basis_set = self._get_basis()

            for i, basis in enumerate(basis_set):
                if gate_pulse:
                    basis = gate_pulse.append(basis, inplace=False)
                if mode=='run':
                    job = self.backend.run(basis, shots=shots)
                    cnt = job.result().get_counts()
                elif mode=='solve':
                    job = self.backend.solve(basis, y0=init_state, t_span=t_span)
                    res_state = job[0].y[-1]
                    cnt = self._get_probs_dict(res_state)
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

            if gate_pulse is not None:
                if t_span is None:
                    t_span = [0., gate_pulse.duration*dt]
                job = self.backend.solve(gate_pulse, y0=init_state, t_span=t_span)
                res_state = job[0].y[-1]
            elif gate_matrix is not None:
                gate_matrix = np.array(gate_matrix)
                if gate_matrix.shape != (3,3):
                    raise Exception(f"The shape {gate_matrix.shape} does not correspond to a single qutrit gate.")
                res_state = np.matmul(gate_matrix, init_state)
            else:
                res_state = init_state

            basis_matrices = self._get_meas_basis_matrix()
            basis_meas_outcomes = {}
            for mat0 in basis_matrices:
                meas_mat = mat0
                fin_state = np.matmul(meas_mat, res_state)
                prob_arr = self._get_probs(fin_state)
                probs.append(prob_arr)  

        eigenvals = self._get_eigenvals()

        coeffs = []
        for prob, ev in zip(probs, eigenvals):
            coeff = 0
            for i in range(3):
                coeff += prob[i]*ev[i]
            coeffs.append(coeff)

        return coeffs
    
    @staticmethod
    def _get_gellmann(idx):
#         c = np.array(3/2, dtype='complex')
#         d = np.array(1/3, dtype='complex')
        
        c = np.array(1, dtype='complex')
        d = np.array(1, dtype='complex')

        gm = [
            d*np.array([[1,0,0],[0,1,0],[0,0,1]]),
            d*c*np.array([[0,1,0],[1,0,0],[0,0,0]]),
            d*c*np.array([[0,-1j,0],[1j,0,0],[0,0,0]]),
            d*c*np.array([[1,0,0],[0,-1,0],[0,0,0]]),
            d*c*np.array([[0,0,1],[0,0,0],[1,0,0]]),
            d*c*np.array([[0,0,-1j],[0,0,0],[1j,0,0]]),
            d*c*np.array([[0,0,0],[0,0,1],[0,1,0]]),
            d*c*np.array([[0,0,0],[0,0,-1j],[0,1j,0]]),
            d*c*np.sqrt(1/3)*np.array([[1,0,0],[0,1,0],[0,0,-2]])
        ]


        return gm[idx]
    
    @staticmethod
    def _get_gellmann_bare(idx):

        gm = [
            np.array([[1,0,0],[0,1,0],[0,0,1]]),
            np.array([[0,1,0],[1,0,0],[0,0,0]]),
            np.array([[0,-1j,0],[1j,0,0],[0,0,0]]),
            np.array([[1,0,0],[0,-1,0],[0,0,0]]),
            np.array([[0,0,1],[0,0,0],[1,0,0]]),
            np.array([[0,0,-1j],[0,0,0],[1j,0,0]]),
            np.array([[0,0,0],[0,0,1],[0,1,0]]),
            np.array([[0,0,0],[0,0,-1j],[0,1j,0]]),
            np.sqrt(1/3)*np.array([[1,0,0],[0,1,0],[0,0,-2]])
        ]


        return gm[idx]
    
    @staticmethod
    def _get_gm_coeff(x):
#         c = np.array(3/2, dtype='complex')
#         d = np.array(1/3, dtype='complex')
        
        c = np.array(1, dtype='complex')
        d = np.array(1, dtype='complex')
        c_list = [d, c*d, c*d, c*d, c*d, c*d, c*d, c*d, c*d]

        return c_list[x]
    
    @staticmethod
    def _commutator(A,B):
        return np.matmul(A,B)-np.matmul(B,A)

    @staticmethod
    def _anticommutator(A,B):
        return np.matmul(A,B)+np.matmul(B,A)

    def _f(self,a,b,c):
        A = self._get_gellmann_bare(a)
        B = self._get_gellmann_bare(b)
        C = self._get_gellmann_bare(c)

        return -(1/4)*1j*np.trace(np.matmul(A, self._commutator(B,C)))

    def _d(self, a,b,c):
        A = self._get_gellmann_bare(a)
        B = self._get_gellmann_bare(b)
        C = self._get_gellmann_bare(c)

        return (1/4)*np.trace(np.matmul(A, self._anticommutator(B,C)))

    def _d_plus_f(self,a,b,c):
        return self._d(a,b,c) + (1j*self._f(a,b,c))
    
    @staticmethod
    def _get_vars(expr):
        syms = expr.free_symbols
        syms_order = []
        for sym in syms:
            syms_order.append((sym, str(expr).index(str(sym))))
        syms_order.sort(key=lambda x: x[1])

        return [sym for (sym,_) in syms_order]

    @staticmethod
    def _get_vars_prod(syms):
        prod = syms[0]
        for sym in syms[1:]:
            prod = prod*sym

        return prod
    
    def get_lambda_identity_process(self):
        
        lambda_matrix = []
        for i in range(9):
            basis_rho = self._get_gellmann(i)
            evals, evecs = np.linalg.eig(basis_rho)
            coeffs_list = []
            for idx, _ in enumerate(evecs):
                    coeffs_list.append(self.run_experiment_for_coeffs(mode='solve', init_state=evecs[:,idx]))
            new_coeffs = []
            for k in range(9):
                s = 0.
                for l in range(3):
                    s = s+(evals[l]*coeffs_list[l][k])
                new_coeffs.append(s)
            lambda_matrix.append(new_coeffs)
        
        lambda_array = []
        for i in range(9):
            for j in range(9):
                lambda_array.append(lambda_matrix[i][j])

        self.lambda_array = np.array(lambda_array)
        
        return self.lambda_array
    
    def get_lambda_process(self):
        if self.gate_pulse is None and self.gate is None:
            print("No Gate not provided. Assuming an identity gate.")
            return self.get_lambda_identity_process()

        lambda_matrix = []
        for i in range(9):
            basis_rho = self._get_gellmann(i)
            evals, evecs = np.linalg.eig(basis_rho)
            coeffs_list = []
            for idx, _ in enumerate(evecs):
                if self.gate_pulse is not None:
                    coeffs_list.append(
                        self.run_experiment_for_coeffs(
                            mode='solve',
                            init_state=evecs[:,idx],
                            gate_pulse=self.gate_pulse
                        )
                    )
                elif self.gate is not None:
                    coeffs_list.append(
                        self.run_experiment_for_coeffs(
                            mode='solve',
                            init_state=evecs[:,idx],
                            gate_matrix=self.gate
                        )
                    )
            new_coeffs = []
            for k in range(9):
                s = 0.
                for l in range(3):
                    s = s+(evals[l]*coeffs_list[l][k])
                new_coeffs.append(s)
            lambda_matrix.append(new_coeffs)
        
        lambda_array = []
        for i in range(9):
            for j in range(9):
                lambda_array.append(lambda_matrix[i][j])

        self.lambda_array = np.array(lambda_array)
        
        return self.lambda_array
        
    def run(self):
        """Returns the process matrix corresponding to the gate."""
        self.lambda_array = self.get_lambda_process()
        chi_array = spsolve(csc_matrix(self.beta_matrix), self.lambda_array)
    
        chi_matrix = []
        for i in range(9):
            chi_matrix.append([])
            for j in range(9):
                chi_matrix[i].append(chi_array[(9*i)+j])
        
        self.chi_matrix = np.array(chi_matrix)
        
        return self.chi_matrix
    
    def plot_chi(self, reals=False, imag=False):
        if reals:
            data = np.real(self.chi_matrix)
        elif imag:
            data = np.imag(self.chi_matrix)
        else:
            data = np.abs(self.chi_matrix)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dims = len(data[0])
        # Add spacing
        spacing = 1.5
        x_idx = np.arange(dims) * spacing
        y_idx = np.arange(dims) * spacing
        x, y = np.meshgrid(x_idx, y_idx)

        # Plot bars
        ax.bar3d(
            x.ravel(), y.ravel(), np.zeros(dims**2), 1, 1, data.ravel(),
            shade=False, alpha=0.5, edgecolor='black', linewidth=0.3
        )


        if dims==3:
            # One-qutrit labels
            labels = [f"|{i}⟩" for i in range(3)]
        elif dims==9:
            # Two-qutrit labels
            labels = [f"|{i}{j}⟩" for i in range(3) for j in range(3)]
        elif dims==81:
            labels = [f"l{i}l{j}" for i in range(9) for j in range(9)]
        else:
            raise Exception(f"Dimensions of data not matching 3 or 9.")

        # Ticks and font size
        ax.set_xticks(x_idx + 0.5)
        ax.set_yticks(y_idx + 0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.tick_params(axis='z', labelsize=8)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        display(plt.show())