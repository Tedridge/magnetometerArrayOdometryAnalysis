import numpy as np
import matplotlib.pyplot as plt
import os as os
from mpl_toolkits.mplot3d import Axes3D

''' Define colours '''
Colours = plt.cm.viridis(np.linspace(1, 0, 3))

''' Create 64 locations for a line '''
line_x = np.linspace(0, 1, 64)
line_y = np.zeros(64)
line_z = np.zeros(64)

''' Create 64 locations for a square '''
square_x = np.linspace(0, 1, 8)
square_y = np.linspace(0, 1, 8)
square_x, square_y = np.meshgrid(square_x, square_y)
square_x = square_x.flatten()
square_y = square_y.flatten()
square_z = np.zeros(64)

''' Create 64 locations for a cube ''' 
cube_x = np.linspace(0, 1, 4)
cube_y = np.linspace(0, 1, 4)
cube_z = np.linspace(0, 1, 4)
cube_x, cube_y, cube_z = np.meshgrid(cube_x, cube_y, cube_z)
cube_x = cube_x.flatten()
cube_y = cube_y.flatten()
cube_z = cube_z.flatten()


''' Create figure '''
fig = plt.figure(figsize=(12, 4))

''' Line plot '''
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(line_x*64, line_y*64, line_z*64, color=Colours[0], label="Line", s=250, edgecolor='black')
ax1.set_xlabel(r"$p_1$", fontsize=20, fontweight='bold')
ax1.set_ylabel(r"$p_2$", fontsize=20, fontweight='bold')
ax1.set_zlabel(r"$p_3$", fontsize=20, fontweight='bold')
ax1.legend(fontsize=16)

''' Square plot '''
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(square_x*8, square_y*8, square_z*8, color=Colours[1], label="Square", s=250, edgecolor='black')
ax2.set_xlabel(r"$p_1$", fontsize=20, fontweight='bold')
ax2.set_ylabel(r"$p_2$", fontsize=20, fontweight='bold')
ax2.set_zlabel(r"$p_3$", fontsize=20, fontweight='bold')
ax2.legend(fontsize=16)
# Draw edges of the square
# for i in range(8):
#     ax2.plot([square_x[i*8], square_x[i*8 + 7]], [square_y[i*8], square_y[i*8 + 7]], [square_z[i*8], square_z[i*8 + 7]], color='grey', linestyle='--')
#     ax2.plot([square_x[i], square_x[i + 56]], [square_y[i], square_y[i + 56]], [square_z[i], square_z[i + 56]], color='grey', linestyle='--')
# for i in range(8):
#     for j in range(7):
#         ax2.plot([square_x[i*8 + j], square_x[i*8 + j + 1]], [square_y[i*8 + j], square_y[i*8 + j + 1]], [square_z[i*8 + j], square_z[i*8 + j + 1]], color='grey', linestyle='--')
#         ax2.plot([square_x[j*8 + i], square_x[j*8 + i + 8]], [square_y[j*8 + i], square_y[j*8 + i + 8]], [square_z[j*8 + i], square_z[j*8 + i + 8]], color='grey', linestyle='--')


''' Cube plot '''
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(cube_x, cube_y, cube_z, color=Colours[2], label="Cube", s=250, edgecolor='black')
ax3.set_xlabel(r"$p_1$", fontsize=20, fontweight='bold')
ax3.set_ylabel(r"$p_2$", fontsize=20, fontweight='bold')
ax3.set_zlabel(r"$p_3$", fontsize=20, fontweight='bold')
ax3.legend(fontsize=16)
# Draw edges of the cube
# for i in range(4):
#     for j in range(4):
#         ax3.plot([cube_x[i*16 + j*4], cube_x[i*16 + j*4 + 3]], [cube_y[i*16 + j*4], cube_y[i*16 + j*4 + 3]], [cube_z[i*16 + j*4], cube_z[i*16 + j*4 + 3]], color='grey', linestyle = '--')
#         ax3.plot([cube_x[i*16 + j], cube_x[i*16 + j + 12]], [cube_y[i*16 + j], cube_y[i*16 + j + 12]], [cube_z[i*16 + j], cube_z[i*16 + j + 12]], color='grey', linestyle = '--')
#         ax3.plot([cube_x[i*4 + j], cube_x[i*4 + j + 48]], [cube_y[i*4 + j], cube_y[i*4 + j + 48]], [cube_z[i*4 + j], cube_z[i*4 + j + 48]], color='grey', linestyle = '--')
# Shared axis labels
fig.text(0.5, -0.05, r"Sensor separation $\alpha_l$ [-]", ha='center', fontsize=20, fontweight='bold')
# fig.text(0.04, 0.5, r"Sensor separation $\alpha/p_2 [
# Adjust layout
# plt.suptitle("3D Line, Square, and Cube in Subplots")
plt.tight_layout()
plt.show()
# Save the figure
# Save the figure
my_path = os.getcwd()
my_data = 'Data'
my_figures = 'Figures'
my_file = 'ConfigurationsExample.pdf'
fig.savefig(os.path.join(my_path, my_figures, my_file), format='pdf', bbox_inches='tight')

