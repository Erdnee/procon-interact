import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

multi_agent_policy1 = np.loadtxt('plot_data_multi_agent1.txt',dtype=float)
multi_agent_policy2 = np.loadtxt('plot_data_multi_agent2.txt',dtype=float)

plt.ylabel("Total Reward")
plt.xlabel("Episode number")

plt.title("Policy gradient on multi agent case")
plt.plot(range(100),multi_agent_policy1[:100], ".", label='agent1')
plt.plot(range(100),multi_agent_policy2[:100], ".", label='agent2')
plt.legend(loc = "upper right")
plt.show()