import matplotlib.pyplot as plt
import numpy as np

def risk_plot(ax, risk_history, soft_th=1, hard_th=2):
    """
    Plots the risk history with a professional appearance.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    risk_history (list or np.array): The history of risk values.
    """
    ax.clear()
    # Plot risk history
    ax.plot(risk_history, label='Risk Level', color='royalblue', linewidth=2)
    # Plot threshold line
    ax.axhline(y=hard_th, color='r', linestyle='-', linewidth=2, label='Hard Threshold')
    ax.axhline(y=soft_th, color='r', linestyle='--', linewidth=2, label='Soft Threshold')
    # Fill above threshold
    ax.fill_between(range(len(risk_history)), risk_history, soft_th, where=(np.array(risk_history) > soft_th),
                    color='red', alpha=0.3)

    # Formatting
    ax.set_title('Risk Level Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=15)
    ax.set_ylabel('Risk Level', fontsize=15)
    ax.set_ylim(0, 3)
    ax.set_xlim(0, len(risk_history))
    ax.legend(loc='upper right', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.pause(0.01)
