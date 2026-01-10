import streamlit as st
import matplotlib.pyplot as plt
from pso import PSO

st.set_page_config(page_title="PSO Ticket Pricing", layout="centered")

st.title("🎬 Cinema Ticket Pricing Optimization")
st.write("Using **Particle Swarm Optimization (PSO)**")

# Sidebar parameters
st.sidebar.header("PSO Parameters")

particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
iterations = st.sidebar.slider("Iterations", 50, 300, 100)
w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.5, 2.5, 1.5)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.5, 2.5, 1.5)

mode = st.sidebar.selectbox(
    "Optimization Mode",
    ["Single Objective (Max Revenue)", "Multi Objective"]
)

alpha = 0.1
if mode == "Multi Objective":
    alpha = st.sidebar.slider(
        "Affordability Weight (alpha)", 0.01, 0.5, 0.1
    )

# Run PSO
pso = PSO(
    n_particles=particles,
    iterations=iterations,
    w=w,
    c1=c1,
    c2=c2,
    multi_objective=(mode == "Multi Objective"),
    alpha=alpha
)

best_price, best_revenue, history = pso.optimize()

# Results
st.subheader("📊 Optimization Results")
st.metric("Optimal Ticket Price (RM)", f"{best_price:.2f}")
st.metric("Maximum Revenue (RM)", f"{best_revenue:.2f}")

# Convergence plot
st.subheader("📈 Convergence Curve")
fig, ax = plt.subplots()
ax.plot(history)
ax.set_xlabel("Iteration")
ax.set_ylabel("Revenue")
ax.grid(True)
st.pyplot(fig)

# Explanation
st.subheader("ℹ️ Explanation")
st.write("""
- **Single Objective**: Maximizes cinema revenue.
- **Multi Objective**: Balances revenue and ticket affordability.
- PSO converges quickly and provides stable pricing solutions.
""")
