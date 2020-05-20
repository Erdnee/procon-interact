import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

area_point_policy1 = np.loadtxt('plot_data_area_point_agent1.txt',dtype=float)
area_point_policy2 = np.loadtxt('plot_data_area_point_agent2.txt',dtype=float)

plt.ylabel("Total Reward")
plt.xlabel("Episode number")

plt.title("Policy gradient on multi agent case")
plt.plot(range(area_point_policy1.size),area_point_policy1, ".", label='agent1')
plt.plot(range(area_point_policy2.size),area_point_policy2, ".", label='agent2')
plt.legend(loc = "upper right")
plt.show()