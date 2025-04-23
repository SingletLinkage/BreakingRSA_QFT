from qiskit import Aer
from qiskit.algorithms import Shor
from qiskit.utils import QuantumInstance

N = 15
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=100)

shor = Shor(quantum_instance=quantum_instance)
result = shor.factor(N)

print(f"Factors of {N}: {result.factors}")