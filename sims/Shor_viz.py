from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from manim import *
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit import QuantumRegister, ClassicalRegister

class ShorsAlgorithm(Scene):
    def construct(self):
        # Title
        title = Text("Shor's Algorithm: Factoring with Quantum Computing", font_size=36).to_edge(UP)
        subtitle = Text("Breaking RSA Encryption with Quantum Computing", font_size=24, color=BLUE).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)
        
        # Introduction
        intro_text = Text("Efficiently factorizes large numbers", font_size=28).next_to(subtitle, DOWN, buff=0.5)
        intro_formula = MathTex(r"N = p \times q", font_size=32).next_to(intro_text, DOWN)
        
        self.play(Write(intro_text))
        self.play(Write(intro_formula))
        self.wait(2)
        
        self.play(FadeOut(subtitle), FadeOut(intro_text), FadeOut(intro_formula))
        
        # Load and display circuit
        circuit_image = ImageMobject('../media/images/shor_circuit.png').scale(0.5)
        circuit_image.to_edge(UP, buff=1.5)
        self.play(FadeIn(circuit_image))
        self.wait(2)
        
        # Circuit explanation
        circuit_parts = VGroup(
            Text("1. Initialization (Hadamard Gates)", font_size=24, color=YELLOW),
            Text("2. Modular Exponentiation", font_size=24, color=GREEN),
            Text("3. Inverse Quantum Fourier Transform", font_size=24, color=RED),
            Text("4. Measurement", font_size=24, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(circuit_image, DOWN, buff=0.5)
        
        for part in circuit_parts:
            self.play(Write(part))
            self.wait(0.5)
        
        self.wait(2)
        
        # Highlight QFT's role
        self.play(FadeOut(circuit_parts))
        qft_text = Text("Quantum Fourier Transform: The Heart of Shor's Algorithm", font_size=28, color=RED).next_to(circuit_image, DOWN)
        self.play(Write(qft_text))
        self.wait(1)
        
        qft_formula = MathTex(r"|j\rangle \mapsto \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} e^{2\pi i \frac{jk}{2^n}} |k\rangle", font_size=28).next_to(qft_text, DOWN)
        self.play(Write(qft_formula))
        self.wait(2)
        
        # Period finding explanation
        self.play(FadeOut(qft_text), FadeOut(qft_formula))
        period_text = Text("Period Finding", font_size=32, color=BLUE).next_to(circuit_image, DOWN)
        period_explain = Tex(r"QFT finds the period $r$ such that $a^r \equiv 1 \pmod{N}$", font_size=28).next_to(period_text, DOWN)
        period_factor = Tex(r"Then factors are $\gcd(a^{r/2} \pm 1, N)$", font_size=28).next_to(period_explain, DOWN)
        
        self.play(Write(period_text))
        self.play(Write(period_explain))
        self.play(Write(period_factor))
        self.wait(3)
        
        # Complexity comparison
        self.play(FadeOut(circuit_image), FadeOut(period_text), FadeOut(period_explain), FadeOut(period_factor))
        
        complexity_title = Text("Algorithm Complexity", font_size=32).to_edge(UP, buff=1.5)
        
        classical = Tex(r"Classical: $O(e^{(\log N)^{1/3}})$", font_size=28, color=RED)
        quantum = Tex(r"Quantum: $O((\log N)^2 \cdot \log \log N)$", font_size=28, color=GREEN)
        
        comparison = VGroup(classical, quantum).arrange(DOWN, buff=0.5).next_to(complexity_title, DOWN, buff=0.5)
        
        self.play(Write(complexity_title))
        self.play(Write(classical))
        self.play(Write(quantum))
        self.wait(2)
        
        # Conclusion
        self.play(FadeOut(complexity_title), FadeOut(comparison))
        
        conclusion = Text("Shor's Algorithm demonstrates quantum advantage", font_size=32, color=YELLOW)
        impact = Text("It threatens RSA encryption and revolutionizes number theory", font_size=28)
        
        conclusion_group = VGroup(conclusion, impact).arrange(DOWN, buff=0.5).center()
        
        self.play(Write(conclusion))
        self.play(Write(impact))
        self.wait(2)
        
        # Fade out everything
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)