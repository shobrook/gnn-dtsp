#########
# GLOBALS
#########


import matplotlib.pyplot as plt
import networkx as nx


#########
# HELPERS
#########


def visualize_network(G, filename, dpi=1000):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "solution")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos)

    plt.savefig("../figures/" + filename, dpi=dpi)
    plt.close()


######
# MAIN
######


# TODO: Load train_stats.pkl and then run this

# Plot results curves
fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)

# Loss
y_tr = losses_tr
y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss (binary cross-entropy)")
ax.legend()

# Correct
y_tr = corrects_tr
y_ge = corrects_ge
ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction correct across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction nodes/edges correct")

# Solved
y_tr = solveds_tr
y_ge = solveds_ge
ax = fig.add_subplot(1, 3, 3)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction solved across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction examples solved")
plt.savefig("../data/figures/train_stats.png", dpi=1000)
