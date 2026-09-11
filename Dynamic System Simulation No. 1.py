import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Step 1: Define System Properties
# ==========================================
# These are the physical parameters of the mass-spring-damper system.
m = 1.0  # Mass (kg)
b = 1.0  # Damping coefficient (N*s/m)
k = 10.0 # Spring constant (N/m)
f = 1.0  # Unit step force applied to the system (N)

# ==========================================
# Step 2: Set Up Simulation Parameters
# ==========================================
dt = 0.01          # Time step (seconds)
t_end = 10.0       # Total simulation time (seconds)

# Create an array of time points from 0 to t_end, spaced by dt
time = np.arange(0, t_end + dt, dt) 

# ==========================================
# Step 3: Initialize State Variables
# ==========================================
# The system starts at rest.
x = 0.0  # Initial position (m)
v = 0.0  # Initial velocity (m/s)

# Create empty lists to store the history of our states for plotting
x_history = []
v_history = []

# ==========================================
# Step 4: Execute First Order Euler Integration
# ==========================================
for t in time:
    # 1. Record current states before updating them
    x_history.append(x)
    v_history.append(v)
    
    # 2. Calculate the derivatives based on the state-space equations
    # This represents the [v_dot, x_dot] matrix calculation
    v_dot = (-b/m) * v - (k/m) * x + (1/m) * f
    x_dot = v
    
    # 3. Apply the Euler Integration step to update the states
    # State = State + Rate * TimeStep
    v = v + v_dot * dt
    x = x + x_dot * dt

# ==========================================
# Step 5: Plot the System Response
# ==========================================
# Create a figure with 2 subplots (Position and Velocity) sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
fig.suptitle('System Response\n2nd Order Spring-Damper System')

# Plot Position (Top Graph)
ax1.plot(time, x_history)
ax1.set_ylabel('Position (m)')
ax1.grid(True)

# Plot Velocity (Bottom Graph)
ax2.plot(time, v_history)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()