import subprocess
import time
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import select

from poolagent.path import VISUALISATIONS_DIR, ROOT_DIR

TIMEOUT = 30

class PoolAnalysisManager:
    def __init__(self, n_processes=8, iterations_per_process=-1, timeout=TIMEOUT, max_restarts=-1):
        self.n_processes = n_processes
        self.iterations_per_process = iterations_per_process if iterations_per_process > 0 else 999999
        self.timeout = timeout
        self.max_restarts = max_restarts if max_restarts > 0 else 999999
        self.save_dir = f"{ROOT_DIR}/poolagent/value_data/logs/stochastic"  
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def run_analysis(self):
        processes = []
        for i in range(self.n_processes):
            process_info = self.start_process(i)
            processes.append(process_info)

        while processes:
            rlist, _, _ = select.select([p[0].stdout for p in processes], [], [], 1.0)
            
            current_time = time.time()
            for i, process_info in enumerate(processes):
                process, start_time, restarts, output_file = process_info
                
                if process.stdout in rlist:
                    output = process.stdout.readline()
                    if output:
                        print(f"Process {i}: {output.decode().strip()}")
                        processes[i] = (process, current_time, restarts, output_file)  # Reset the timer
                
                if process.poll() is not None:
                    print(f"Process {i} completed.")
                    processes.pop(i)
                    break
                
                if time.time() - start_time > self.timeout:
                    print(f"Process {i} timed out.")
                    process.terminate()
                    if restarts < self.max_restarts:
                        print(f"Restarting process {i}...")
                        new_process_info = self.start_process(i, restarts + 1)
                        processes[i] = new_process_info
                    else:
                        print(f"Process {i} exceeded maximum restarts. Removing from pool.")
                        processes.pop(i)
                    break

    def start_process(self, process_index, restarts=0):
        output_file = f"{self.save_dir}/stochastic_data_p{process_index}_r{restarts}_{self.timestamp}.json"
        process = subprocess.Popen(
            ['python', '-m' 'poolagent.value_data.gen_stochastic_data', 
             str(self.iterations_per_process), 
             output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return (process, time.time(), restarts, output_file)

    @staticmethod
    def plot_shot_quality_estimates(estimates, initial_estimate):
        mean = np.mean(estimates)
        std = np.std(estimates)
        iqr = float(stats.iqr(estimates))

        fig, ax = plt.subplots(figsize=(12, 6))

        kde = stats.gaussian_kde(estimates)
        x_range = np.linspace(min(estimates), max(estimates), 100)
        pdf = kde(x_range)

        ax.plot(x_range, pdf, label='Estimates PDF', color='blue')
        ax.fill_between(x_range, pdf, alpha=0.3, color='blue')

        ax.axvline(initial_estimate, label='Initial Estimate', color='green', linestyle='--')

        norm_dist = stats.norm(mean, std)
        ax.plot(x_range, norm_dist.pdf(x_range), label='Normal Distribution', color='red', linestyle='--')

        ax.set_xlabel('Value Estimate')
        ax.set_ylabel('Probability Density')
        ax.set_title('Pool Shot Quality Estimates Distribution')
        
        ax.text(0.05, 0.95, f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\nIQR: {iqr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{VISUALISATIONS_DIR}/shot_quality_estimates.png")

if __name__ == "__main__":
    manager = PoolAnalysisManager()
    manager.run_analysis()
