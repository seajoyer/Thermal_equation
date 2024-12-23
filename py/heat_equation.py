import numpy as np

class HeatEquationSolver:
    def __init__(self,
                 x_start: float = 0.0,
                 x_end: float = 1.0,
                 t_end: float = 1.0,
                 dx: float = 0.05,
                 dt: float = 0.05):
        self.x_start = x_start
        self.x_end = x_end
        self.t_end = t_end
        self.dx = dx
        self.dt = dt

        # grid points
        self.nx = int((x_end - x_start) / dx) + 1
        self.nt = int(t_end / dt) + 1

        self.x = np.linspace(x_start, x_end, self.nx)
        self.t = np.linspace(0, t_end, self.nt)

    def _thomas_algorithm(self, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(d)
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        x = np.zeros(n)

        # Forward
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n-1):
            denominator = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denominator
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator

        d_prime[n-1] = (d[n-1] - a[n-1] * d_prime[n-2]) / (b[n-1] - a[n-1] * c_prime[n-2])

        # Back
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def solve(self, order: int = 2) -> np.ndarray:
        # solution array
        u = np.zeros((self.nt, self.nx))

        # initial condition
        u[0] = np.cosh(self.x)

        # Initialize matrices
        a = np.zeros(self.nx)
        b = np.zeros(self.nx)
        c = np.zeros(self.nx)
        d = np.zeros(self.nx)

        # Time stepping
        for n in range(self.nt-1):
            if order == 1:
                # First-order scheme
                r = self.dt / (self.dx**2)

                for i in range(1, self.nx-1):
                    a[i] = -r
                    b[i] = 1 + 2*r
                    c[i] = -r
                    d[i] = u[n,i] + self.dt * (
                        np.sinh(self.x[i] - self.t[n]) +
                        np.cosh(self.x[i] - self.t[n])
                    )

            else:
                # Second-order scheme
                r = self.dt / (2 * self.dx**2)

                for i in range(1, self.nx-1):
                    a[i] = -r
                    b[i] = 1 + 2*r
                    c[i] = -r
                    d[i] = (r * u[n,i-1] +
                           (1 - 2*r) * u[n,i] +
                           r * u[n,i+1] +
                           self.dt * (
                               np.sinh(self.x[i] - self.t[n]) +
                               np.cosh(self.x[i] - self.t[n])
                           ))

            # Left boundary
            b[0] = 1 + 1/self.dx
            c[0] = -1/self.dx
            a[0] = 0
            d[0] = np.exp(self.t[n+1])

            # Right boundary
            a[-1] = -1/self.dx
            b[-1] = 1 + 1/self.dx
            c[-1] = 0
            d[-1] = np.exp(1 - self.t[n+1])

            # Solve system
            u_new = self._thomas_algorithm(a, b, c, d)

            if np.any(np.isnan(u_new)) or np.any(np.abs(u_new) > 1e10):
                raise RuntimeError("Solution becoming unstable")

            u[n+1] = u_new

        return u

    def get_exact_solution(self) -> np.ndarray:
        u_exact = np.zeros((self.nt, self.nx))
        for n in range(self.nt):
            u_exact[n] = np.cosh(self.x - self.t[n])
        return u_exact
