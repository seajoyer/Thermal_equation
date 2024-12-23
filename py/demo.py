import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heat_equation import HeatEquationSolver

def plot_solutions(solver: HeatEquationSolver):
    try:
        u1 = solver.solve(order=1)
        u2 = solver.solve(order=2)
        u_exact = solver.get_exact_solution()

        error1 = np.abs(u1 - u_exact)
        error2 = np.abs(u2 - u_exact)

        plt.style.use('default')
        fig = plt.figure(figsize=(12, 15))

        vmin = np.min(u_exact)
        vmax = np.max(u_exact)

        # first-order solution
        ax1 = plt.subplot(321)
        im1 = ax1.pcolormesh(solver.x, solver.t, u1, shading='auto',
                            cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('First-order Numerical Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')

        # second-order solution
        ax2 = plt.subplot(322)
        im2 = ax2.pcolormesh(solver.x, solver.t, u2, shading='auto',
                            cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Second-order Numerical Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')

        # first-order error
        ax3 = plt.subplot(323)
        im3 = ax3.pcolormesh(solver.x, solver.t, error1, shading='auto',
                            cmap='viridis')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title(f'First-order Error (Max: {np.max(error1):.3e})')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')

        # second-order error
        ax4 = plt.subplot(324)
        im4 = ax4.pcolormesh(solver.x, solver.t, error2, shading='auto',
                            cmap='viridis')
        plt.colorbar(im4, ax=ax4)
        ax4.set_title(f'Second-order Error (Max: {np.max(error2):.3e})')
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')

        ax5 = plt.subplot(313)

        ax5.set_xlim(0, 1)
        ymin = min(np.min(u1), np.min(u2), np.min(u_exact))
        ymax = max(np.max(u1), np.max(u2), np.max(u_exact))
        padding = 0.1 * (ymax - ymin)
        ax5.set_ylim(ymin - padding, ymax + padding)

        line1, = ax5.plot([], [], 'b-', label='First-order', linewidth=2)
        line2, = ax5.plot([], [], 'g-', label='Second-order', linewidth=2)
        line3, = ax5.plot([], [], 'r--', label='Exact', linewidth=2)
        time_text = ax5.text(0.02, 0.95, '', transform=ax5.transAxes)

        ax5.set_xlabel('x')
        ax5.set_ylabel('u(x,t)')
        ax5.grid(True)
        ax5.legend()

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            time_text.set_text('')
            return line1, line2, line3, time_text

        def animate(frame):
            t = solver.t[frame]
            line1.set_data(solver.x, u1[frame])
            line2.set_data(solver.x, u2[frame])
            line3.set_data(solver.x, u_exact[frame])
            time_text.set_text(f't = {t:.2f}')
            ax5.set_title(f'Solution comparison at t = {t:.2f}')
            return line1, line2, line3, time_text

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(solver.t), interval=50,
                                     blit=True, repeat=True)

        plt.tight_layout()

        anim.save('heat_equation_animation.gif', writer='pillow', fps=20)

        plt.show()

    except RuntimeError as e:
        print(f"Error during solution: {e}")

def main():
    solver = HeatEquationSolver(
        x_start=0.0,
        x_end=1.0,
        t_end=1.0,
        dx=0.05,
        dt=0.01
    )

    plot_solutions(solver)

if __name__ == "__main__":
    main()
