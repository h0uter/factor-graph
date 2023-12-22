import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Set the mean and covariance matrix
mean = np.array([2, 3])
covariance_matrix = np.array([[1, 0.5], [0.5, 2]])

# Generate a grid of points
x, y = np.meshgrid(np.linspace(-1, 5, 100), np.linspace(-1, 7, 100))
pos = np.dstack((x, y))

# Create a multivariate normal distribution
rv = multivariate_normal(mean, covariance_matrix)

# Calculate the probability density function (pdf) values for each point in the grid
z = rv.pdf(pos)

# Plot the contour plot
plt.contour(x, y, z, levels=10, cmap="viridis")


# def create_cov_elipse_actor(
#     covariance_matrix: np.ndarray, mean: np.ndarray = np.zeros(2)
# ):
#     # Plot the covariance matrix as ellipses
#     v, w = np.linalg.eigh(covariance_matrix)
#     angle = np.arctan2(w[0, 1], w[0, 0])
#     angle = 180 * angle / np.pi  # convert to degrees
#     v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

#     ell = patches.Ellipse(mean, v[0], v[1], angle=180 + angle, color="red", alpha=0.3)
#     return ell


# ell = create_cov_elipse_actor(covariance_matrix, mean)

# plt.gca().add_patch(ell)

# Add a scatter plot of the mean
plt.scatter(mean[0], mean[1], color="red", marker="x", label="Mean")

# Set labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D Gaussian Distribution with Covariance Matrix Ellipse")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Probability Density")

# Add legend
plt.legend()

# Show the plot
plt.show()
