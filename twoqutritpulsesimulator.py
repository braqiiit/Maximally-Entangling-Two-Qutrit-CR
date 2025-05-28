import numpy as np
from qiskit import pulse
from qiskit_dynamics import Solver, DynamicsBackend 
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate, HGate
from qiskit.circuit import Parameter, Gate
from qiskit.providers.backend import QubitProperties
from qiskit.transpiler import InstructionProperties
from qiskit.providers.models import PulseDefaults


class TwoQutritPulseSimulator():
    def __init__(
        self,
        qubit_frequency_0 = 4.9e9,
        qubit_frequency_1 = 5.5e9,
        anharmonicity_0 = -0.4e9,
        anharmonicity_1 = -0.3e9,
        drive_strength_0 = 0.62e9,
        drive_strength_1 = 0.64e9,
    ):
        dim = 3

        self.v0 = qubit_frequency_0
        self.anharmon_0 = anharmonicity_0
        self.r0 = drive_strength_0
        
        self.v1 = qubit_frequency_1
        self.anharmon_1 = anharmonicity_1
        self.r1 = drive_strength_1
        
        self.J = 2.7e6
        
        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
        N = np.diag(np.arange(dim))
        
        ident = np.eye(dim, dtype=complex)
        full_ident = np.eye(dim**2, dtype=complex)
        
        N0 = np.kron(ident, N)
        N1 = np.kron(N, ident)
        
        a0 = np.kron(ident, a)
        a1 = np.kron(a, ident)
        
        a0dag = np.kron(ident, adag)
        a1dag = np.kron(adag, ident)
        
        
        static_ham0 = 2 * np.pi * self.v0 * N0 + np.pi * self.anharmon_0 * N0 * (N0 - full_ident)
        static_ham1 = 2 * np.pi * self.v1 * N1 + np.pi * self.anharmon_1 * N1 * (N1 - full_ident)
        
        static_ham_full = static_ham0 + static_ham1 + 2 * np.pi * self.J * ((a0 + a0dag) @ (a1 + a1dag))
        
        drive_op0 = 2 * np.pi * self.r0 * (a0 + a0dag)
        drive_op1 = 2 * np.pi * self.r1 * (a1 + a1dag)

        # build solver
        dt = 1/4.5e9
        
        self.solver = Solver(
            static_hamiltonian=static_ham_full,
            hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],
            # rotating_frame=static_ham_full,
            hamiltonian_channels=["d0", "d1", "u0", "u1"],
            channel_carrier_freqs={"d0": self.v0, "d1": self.v1, "u0": self.v1, "u1": self.v0},
            dt=dt,
            array_library="jax",
        )

        # Consistent solver option to use throughout notebook
        solver_options = {"method": "jax_odeint", "atol": 1e-11, "rtol": 1e-11, "hmax": dt}
        
        self.backend = DynamicsBackend(
            solver=self.solver,
            subsystem_dims=[dim, dim], # for computing measurement data
            solver_options=solver_options, # to be used every time run is called
            max_outcome_level=dim,
        )

        self.target = self.backend.target

        # qubit properties
        self.target.qubit_properties = [QubitProperties(frequency=self.v0), QubitProperties(frequency=self.v1)]
        
        #####################################################################################
        ##### add instructions for qutrits #####

        x01_gate = Gate('x01', 1, [])
        
        with pulse.build() as x01_0:
            pulse.Play(pulse.Gaussian(320, 0.5, 80), pulse.DriveChannel(0))
        
        with pulse.build() as x01_1:
            pulse.Play(pulse.Gaussian(320, 0.5, 80), pulse.DriveChannel(1))

        self.target.add_instruction(
            x01_gate,
            {
                (0,): InstructionProperties(calibration=x01_0), 
                (1,): InstructionProperties(calibration=x01_1)
            }
        )
        
        self.target.add_instruction(CXGate(), properties={(0, 1): None, (1, 0): None})
        
        #### Add RZ instruction as phase shift for drag cal  #####
        phi = Parameter("phi")
        with pulse.build() as rz0:
            pulse.shift_phase(phi, pulse.DriveChannel(0))
            pulse.shift_phase(phi, pulse.ControlChannel(1))
        
        with pulse.build() as rz1:
            pulse.shift_phase(phi, pulse.DriveChannel(1))
            pulse.shift_phase(phi, pulse.ControlChannel(0))
        
        self.target.add_instruction(
            RZGate(phi),
            {(0,): InstructionProperties(calibration=rz0), (1,): InstructionProperties(calibration=rz1)}
        )

        ##################################################################################

        self.backend.set_options(control_channel_map={(0, 1): 0, (1, 0): 1})

    def get_backend(self):
        return self.backend

    def get_solver(self):
        return self.solver

    def get_target(self):
        return self.target