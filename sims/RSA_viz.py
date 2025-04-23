from manim import *

class RSAFullDemo(Scene):
    def construct(self):
        # --- Title ---
        title = Text("RSA Key Generation").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # --- Key Generation Steps ---
        step1 = Tex(r"1. Choose two large primes \( p \) and \( q \)").shift(UP * 2).align_to(ORIGIN, RIGHT)
        step2 = Tex(r"2. Compute \( N = p \cdot q \)").next_to(step1, DOWN, aligned_edge=LEFT)
        step3 = Tex(r"3. Compute \( \phi(N) = (p-1) \cdot (q-1) \)").next_to(step2, DOWN, aligned_edge=LEFT)
        step4 = Tex(r"4. Choose \( e \) such that \( 1 < e < \phi(N) \), \( \gcd(e, \phi(N)) = 1 \)").next_to(step3, DOWN, aligned_edge=LEFT)
        step5 = Tex(r"5. Compute \( d \) such that \( d \cdot e \equiv 1 \mod \phi(N) \)").next_to(step4, DOWN, aligned_edge=LEFT)

        # Group all steps and center them horizontally
        steps_group = VGroup(step1, step2, step3, step4, step5).center().shift(LEFT * 0.5)
        
        # Now display each step
        for step in [step1, step2, step3, step4, step5]:
            self.play(Write(step))
            self.wait(1)

        self.play(FadeOut(Group(*self.mobjects)))  # Fade out everything
        self.wait(1)

        # --- Encryption & Decryption ---
        title2 = Text("RSA Encryption and Decryption").to_edge(UP)
        self.play(Write(title2))
        self.wait(1)

        enc = Tex(r"Encrypt: \( c = m^e \mod N \)").shift(UP)
        dec = Tex(r"Decrypt: \( m = c^d \mod N \)").next_to(enc, DOWN)

        self.play(Write(enc))
        self.wait(3)
        self.play(Write(dec))
        self.wait(3)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)

        # --- Practical Example ---
        title3 = Text("RSA Example (p=5, q=11)").to_edge(UP)
        self.play(Write(title3))
        self.wait(1)

        example_steps = [
            r"1. \( p = 5, \ q = 11 \Rightarrow N = 55 \)",
            r"2. \( \phi(N) = (5-1)(11-1) = 4 \cdot 10 = 40 \)",
            r"3. Choose \( e = 3 \), since \( \gcd(3, 40) = 1 \)",
            r"4. Compute \( d \) such that \( 3d \equiv 1 \mod 40 \Rightarrow d = 27 \)",
            r"5. Public key: \( (e=3, N=55) \), Private key: \( (d=27, N=55) \)",
            r"6. Encrypt message \( m = 12 \): \( c = 12^3 \mod 55 = 1728 \mod 55 = 23 \)",
            r"7. Decrypt: \( m = 23^{27} \mod 55 = 12 \)"
        ]

        # Create all example steps first
        example_mobs = VGroup(*[Tex(eq).scale(0.85) for eq in example_steps])
        # Arrange them in a column with proper alignment
        example_mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.4).center()

        # Display each step
        for tex in example_mobs:
            self.play(Write(tex))
            self.wait(2)

        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

        outro = Text("That's how RSA works! 🔐").scale(1.2)
        self.play(Write(outro))
        self.wait(3)

if __name__ == "__main__":
    from manim import config, tempconfig

    with tempconfig({
        "quality": "production_quality",  # Choose: low_quality, medium_quality, high_quality, production_quality
        "preview": True,
        "output_file": "rsa_full_demo",
    }):
        scene = RSAFullDemo()
        scene.render()
