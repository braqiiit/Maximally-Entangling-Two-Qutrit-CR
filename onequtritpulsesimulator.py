import numpy as np
from qiskit import pulse
from qiskit_dynamics import Solver, DynamicsBackend  
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate, HGate
from qiskit.circuit import Parameter, Gate
from qiskit.providers.backend import QubitProperties
from qiskit.transpiler import InstructionProperties
from qiskit.providers.models import PulseDefaults


class OneQutritPulseSimulator():
    def __init__(
        self,
        qubit_frequency = 5e9,
        anharmonicity = -0.3e9,
        drive_strength = 0.22e9,
    ):
        dim = 3

        self.v0 = qubit_frequency
        self.anharmon_0 = anharmonicity
        self.r0 = drive_strength
        
        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
        N = np.diag(np.arange(dim))
        
        ident = np.eye(dim, dtype=complex)
        
        
        static_ham0 = 2 * np.pi * self.v0 * N + np.pi * self.anharmon_0 * N * (N - ident)
        
        static_ham_full = static_ham0
        
        drive_op0 = 2 * np.pi * self.r0 * (a + adag)

        # build solver
        dt = 1/4.5e9
        
        self.solver = Solver(
            static_hamiltonian=static_ham_full,
            hamiltonian_operators=[drive_op0],
            rotating_frame=static_ham_full,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": self.v0},
            dt=dt,
            array_library="jax",
        )

        # Consistent solver option to use throughout notebook
        solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8, "hmax": dt}
        
        self.backend = DynamicsBackend(
            solver=self.solver,
            subsystem_dims=[dim], # for computing measurement data
            solver_options=solver_options, # to be used every time run is called
            max_outcome_level=dim,
        )

        self.target = self.backend.target

        # qubit properties
        self.target.qubit_properties = [QubitProperties(frequency=self.v0)]
        

        x01_gate = Gate('x01', 1, [])
        
        with pulse.build() as x01_0:
            pulse.Play(pulse.Gaussian(320, 0.5, 80), pulse.DriveChannel(0))

        self.target.add_instruction(
            x01_gate,
            {
                (0,): InstructionProperties(calibration=x01_0)
            }
        )

    def get_backend(self):
        return self.backend

    def get_solver(self):
        return self.solver

    def get_target(self):
        return self.target