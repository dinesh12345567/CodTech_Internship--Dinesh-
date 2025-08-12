# Import PuLP
from pulp import LpMaximize, LpProblem, LpVariable, value, LpStatus

# Create a Linear Programming model for maximization
lp_model = LpProblem("Profit_Optimization", LpMaximize)

# Decision variables: quantities of Product A and B
A = LpVariable("Units_A", lowBound=0)
B = LpVariable("Units_B", lowBound=0)

# Objective Function: Maximize profit
lp_model += 30 * A + 50 * B, "Profit"

# Constraints
lp_model += A + 2 * B <= 40, "Machine_Usage"
lp_model += 2 * A + 1 * B <= 60, "Labor_Usage"

# Solve the LP
lp_model.solve()

# Results
print(f"Status: {LpStatus[lp_model.status]}")
print(f"Optimal production → Product A: {A.varValue:.2f} units")
print(f"Optimal production → Product B: {B.varValue:.2f} units")
print(f"Maximum Profit: ₹{value(lp_model.objective):.2f}")

