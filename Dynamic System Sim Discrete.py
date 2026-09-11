import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv

# ==========================================
# Step 1: Define System Properties
# ==========================================
m = 1.0  # Mass (kg)
b = 1.0  # Damping coefficient (N*s/m)
k = 10.0 # Spring constant (N/m)
f = 1.0  # Unit step force (N)

dt = 0.1         # Time step (seconds) - Notice this is 10x larger than the Euler method!
t_end = 10.0     # Total simulation time (seconds)

# ==========================================
# Step 2: Define Continuous-Time Matrices
# ==========================================
# The continuous state-space equation is: X_dot = A * X + B * u
# Here we define the A and B matrices from the system diagram.
A = np.array([
    [-b/m, -k/m],
    [ 1.0,  0.0]
])

B = np.array([
    [1/m],
    [0.0]
])

# ==========================================
# Step 3: Matrix Exponential Discretisation
# ==========================================
# To convert the continuous system to a discrete system (X[k+1] = Ad * X[k] + Bd * u[k]),
# we use the matrix exponential method (assuming a Zero-Order Hold on the input).

# Calculate the discrete state matrix (Ad = e^(A*dt))
Ad = expm(A * dt)

# Calculate the discrete input matrix (Bd = A^-1 * (Ad - I) * B)
I = np.eye(2)       # Creates a 2x2 Identity matrix
A_inv = inv(A)      # Calculates the inverse of matrix A
Bd = A_inv @ (Ad - I) @ B

# ==========================================
# Step 4: Initialize Simulation
# ==========================================
time = np.arange(0, t_end + dt, dt)

# Define the initial state as a 2x1 column vector: [velocity; position]
X = np.array([
    [0.0], 
    [0.0]
]) 

v_history = []
x_history = []

# ==========================================
# Step 5: Execute Discrete Simulation
# ==========================================
for t in time:
    # Record current state. X[0,0] is velocity, X[1,0] is position.
    v_history.append(X[0, 0])
    x_history.append(X[1, 0])
    
    # Calculate the exact state at the next time step.
    # The '@' operator performs matrix multiplication in Python.
    X = Ad @ X + Bd * f

# ==========================================
# Step 6: Plot the System Response
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
fig.suptitle('System Response\n2nd Order Spring-Damper System (Discrete)')

ax1.plot(time, x_history)
ax1.set_ylabel('Position (m)')
ax1.grid(True)

ax2.plot(time, v_history)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True)

plt.tight_layout()
plt.show()