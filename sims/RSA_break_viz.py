from manim import *

class BreakingRSA(Scene):
    def construct(self):
        # Title with styling
        title = Text("Breaking RSA with Shor's Algorithm", font="Arial", color=BLUE_C).to_edge(UP)
        subtitle = Text("Quantum Computing's Threat to Classical Cryptography", font="Arial",
                         color=RED, font_size=24).next_to(title, DOWN, buff=0.3)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP*0.2))
        self.wait(1)

        # First create the RSA content larger and centered
        rsa_title = Text("RSA Encryption", font_size=42, color=GREEN_A)
        rsa_content = VGroup(
            Tex(r"Public Key: $(N, e)$", font_size=36),
            Tex(r"Private Key: $d$", font_size=36),
            Tex(r"$N = p \times q$ (large primes)", font_size=36)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        rsa_group_content = VGroup(rsa_title, rsa_content).arrange(DOWN, buff=0.3).center()
        
        # Show it centered first
        self.play(FadeIn(rsa_group_content))
        self.wait(1)
        
        # Create a surrounding rectangle for the initial centered content
        initial_rsa_box = SurroundingRectangle(rsa_group_content, buff=0.3, color=GREEN)
        rsa_group_initial = VGroup(initial_rsa_box, rsa_group_content)
        
        self.play(Create(initial_rsa_box))
        self.wait(1)
        
        # Create the final version (smaller and positioned differently)
        rsa_title_final = Text("RSA Encryption", font_size=30, color=GREEN_A)
        rsa_content_final = VGroup(
            Tex(r"Public Key: $(N, e)$", font_size=22),
            Tex(r"Private Key: $d$", font_size=22),
            Tex(r"$N = p \times q$ (large primes)", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rsa_group_content_final = VGroup(rsa_title_final, rsa_content_final).arrange(DOWN, buff=0.2)
        
        # Create the final surrounding rectangle
        final_rsa_box = SurroundingRectangle(rsa_group_content_final, buff=0.2, color=GREEN)
        
        # Position the final group in the desired location
        final_group = VGroup(final_rsa_box, rsa_group_content_final).to_corner(DL, buff=1)
        
        # Transform everything together to avoid unwanted fade-in
        self.play(
            Transform(rsa_group_initial, final_group),
            run_time=1.5
        )
        
        # Redefine rsa_group for future reference
        rsa_group = rsa_group_initial
        
        self.wait(2)
        
        # Step 1: Factor N
        step1_title = Text("Step 1: Quantum Factorization", font_size=28, color=YELLOW_C).to_edge(UP, buff=2)
        
        num_example = VGroup(
            Tex(r"Example: $N = 15$", font_size=26),
            Tex(r"Shor's algorithm finds:", font_size=26),
            Tex(r"$p = 3, q = 5$", font_size=26, color=YELLOW)
        ).arrange(DOWN, buff=0.3).next_to(step1_title, DOWN, buff=0.5)
        
        # Visual representation of factorization
        factor_arrow = Arrow(LEFT, RIGHT, color=YELLOW).next_to(num_example, DOWN, buff=0.5)
        number_15 = MathTex("15", font_size=30).next_to(factor_arrow, LEFT, buff=0.5)
        factors = MathTex("3 \\times 5", font_size=30, color=YELLOW).next_to(factor_arrow, RIGHT, buff=0.5)
        
        self.play(Write(step1_title))
        self.play(Write(num_example))
        self.play(Create(factor_arrow), Write(number_15), Write(factors))
        self.wait(2)
        
        step1_group = VGroup(step1_title, num_example, factor_arrow, number_15, factors)
        
        # Step 2: Compute φ(N)
        step2_title = Text("Step 2: Calculate Euler's Totient", font_size=28, color=RED_C)
        phi_calc = VGroup(
            Tex(r"$\phi(N) = (p-1)(q-1)$", font_size=28),
            Tex(r"$\phi(15) = (3-1)(5-1)$", font_size=28),
            Tex(r"$\phi(15) = 2 \times 4 = 8$", font_size=28, color=RED)
        ).arrange(DOWN, buff=0.3)
        
        step2_group = VGroup(step2_title, phi_calc).arrange(DOWN, buff=0.4)
        
        self.play(FadeOut(step1_group))
        self.play(Write(step2_title))
        self.play(Write(phi_calc))
        self.wait(2)

        
        # Step 3: Computing d
        step3_title = Text("Step 3: Find Private Key d", font_size=28, color=PURPLE_C)
        key_calc_lines = [
            Tex(r"$d \cdot e \equiv 1 \mod \phi(N)$", font_size=28),
            Tex(r"Example: if $e = 3$", font_size=28),
            Tex(r"$d \cdot 3 \equiv 1 \mod 8$", font_size=28),
            Tex(r"$d = 3$ is the solution", font_size=28, color=PURPLE)
        ]
        
        key_calc = VGroup(*key_calc_lines).arrange(DOWN, buff=0.3)
        
        # Create the step3_group first without the box
        step3_group = VGroup(step3_title, key_calc).arrange(DOWN, buff=0.4)
        
        # Add the step3_group to the scene first, then create the box after rendering
        self.play(FadeOut(step2_group))
        self.play(Write(step3_title))
        self.play(Write(key_calc))
        # Ensure the mobject is rendered before creating the box
        self.wait(0.1)
        d_box = SurroundingRectangle(key_calc_lines[3], buff=0.15, color=PURPLE, stroke_width=2)
        self.play(Create(d_box))
        self.wait(2)
        
        # For the fadeout, include both the step3_group and the d_box
        step3_group = VGroup(step3_title, key_calc, d_box)
        
        # Step 4: Decrypt
        step4_title = Text("Step 4: Decrypt Messages", font_size=28, color=TEAL_C)
        decrypt_example = VGroup(
            Tex(r"Ciphertext: $C = M^e \mod N$", font_size=28),
            Tex(r"Decryption: $M = C^d \mod N$", font_size=28),
            Tex(r"With private key $d$, all messages can be decrypted", font_size=28, color=TEAL)
        ).arrange(DOWN, buff=0.3)
        
        # Create a simple lock icon using basic shapes
        lock_base = Circle(radius=0.3, color=TEAL, fill_opacity=0.3)
        lock_body = Rectangle(height=0.5, width=0.4, color=TEAL, fill_opacity=0.3).next_to(lock_base, DOWN, buff=0)
        lock_open = VGroup(lock_base, lock_body).scale(0.5)
        
        step4_group = VGroup(step4_title, decrypt_example, lock_open).arrange(DOWN, buff=0.4)
        
        self.play(FadeOut(step3_group))
        self.play(Write(step4_title))
        self.play(Write(decrypt_example))
        self.play(DrawBorderThenFill(lock_open))
        self.wait(2)
        
        # Conclusion
        conclusion = VGroup(
            Text("Implications for Cryptography", font_size=32, color=RED),
            Tex(r"• Quantum computers threaten RSA security", font_size=24),
            Tex(r"• Need for quantum-resistant cryptography", font_size=24),
            Tex(r"• Estimated timeline: 5-15+ years", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        self.play(FadeOut(step4_group), FadeOut(rsa_group))
        self.play(FadeIn(conclusion))
        self.wait(2)

        # Final fade out
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)
