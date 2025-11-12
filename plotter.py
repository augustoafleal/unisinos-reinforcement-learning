from util.plots import plot_learning_curve_workers

plot_learning_curve_workers(f"logs/worker_logs_ppo_random_12x12_1.csv", window_size=20)
# plot_learning_curve_workers(f"logs/worker_logs_ppo_4x4_1.csv", window_size=20)
