from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

def shors_circuit():
    # For factoring N = 15, using a = 2
    # 4 counting qubits, 4 work qubits
    qc = QuantumCircuit(8, 4)
    
    # Apply Hadamard to counting qubits (0-3)
    qc.h(range(4))
    
    # Initialize work register (qubits 4-7) to |1>
    qc.x(4)
    
    qc.barrier()

    # Controlled-U^{2^0}
    qc.cswap(0, 4, 5)
    qc.cswap(0, 5, 6)
    qc.cswap(0, 6, 7)
    
    # Controlled-U^{2^1}
    qc.cswap(1, 4, 5)
    qc.cswap(1, 5, 6)
    qc.cswap(1, 6, 7)

    # Controlled-U^{2^2}
    qc.cswap(2, 4, 5)
    qc.cswap(2, 5, 6)
    qc.cswap(2, 6, 7)

    # Controlled-U^{2^3}
    qc.cswap(3, 4, 5)
    qc.cswap(3, 5, 6)
    qc.cswap(3, 6, 7)
    
    qc.barrier()

    # Apply inverse QFT
    qc.append(QFT(4).inverse(), [0, 1, 2, 3])
    
    qc.barrier()
    
    # Measure the counting register
    qc.measure(range(4), range(4))

    return qc

# Save circuit as image
qc = shors_circuit()
qc.draw(output='mpl', filename='../media/images/shor_circuit.png')