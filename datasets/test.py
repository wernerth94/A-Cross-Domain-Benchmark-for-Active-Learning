import numpy as np
import matplotlib.pyplot as plt

num = 100
cov=.3
fact = 0.5
# Define the x-values for the sine curve
x = np.linspace(0, 10, num)


cov = .3
# Generate the sine curve
sin_curve = np.sin(2*x)

# Generate a single cluster above and below the sine curve
np.random.seed(42)  # Set a seed for reproducibility

# Cluster above the curve
cluster_above_x = x
cluster_above_y = sin_curve + fact * np.abs(x) + np.random.normal(0, cov , num)

# Cluster below the curve
cluster_below_x = x
cluster_below_y = sin_curve - fact * np.abs(x) + np.random.normal(0, cov, num)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, sin_curve, color='blue', label='Sine Curve')
plt.scatter(cluster_above_x, cluster_above_y, color='red', label='Cluster Above')
plt.scatter(cluster_below_x, cluster_below_y, color='green', label='Cluster Below')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Single Cluster Above and Below the Sine Curve')
plt.grid(True)
plt.show()

print(cluster_below_x)
print(cluster_below_y)