def fermat_factorization(n):

    # Start with the ceiling of sqrt(n)
    a = ceil(sqrt(n))
    
    # Initialize b_squared = a^2 - n
    b_squared = a^2 - n
    
    # Continue until we find a perfect square
    while not is_square(b_squared):
        a += 1
        b_squared = a^2 - n
    
    # Once we find a perfect square, compute b
    b = sqrt(b_squared)
    
    # Return the factors
    return (a - b, a + b)

n = 5959
p, q = fermat_factorization(n)
print(f"Factors of {n}: {p} and {q}")
print(f"Verification: {p} * {q} = {p*q}")