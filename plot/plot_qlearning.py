import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

q_table = np.loadtxt('plot_data_qtable.txt', dtype = float)
q_table_dynamic_reward = np.loadtxt('plot_data_qtable_with_dynamic_reward.txt',dtype=float)

for i in range(q_table.size):
    if(q_table[i] == 0):
        q_table[i] = q_table[i-1]

plt.ylabel("Total Reward")
plt.xlabel("Episode number")
plt.title("Q_learning plot")
plt.plot(range(5000),q_table_dynamic_reward[:5000],".", label='non stationary environment')
plt.plot(range(5000),q_table[:5000],".",label = 'stationary environment')
plt.legend(loc = "lower right")
plt.show()
plt.savefig("qlearning_plot.png")