class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        # val = init**2
        grad = 0.0
        for i in range(1,iterations+1):
            grad = 2*init
            init = init-learning_rate*grad
            
        return round(init,5)
