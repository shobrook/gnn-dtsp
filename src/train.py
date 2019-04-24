#########
# GLOBALS
#########


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle

from graph_nets.demos import models


#########
# HELPERS
#########


def create_loss_ops(target_op, output_ops):
    return [
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")

    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        # xn = np.argmax(td["nodes"], axis=-1)
        # yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        # if use_nodes:
        #     c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))

    return correct, solved


#############
# MODEL SETUP
#############


tf.reset_default_graph()

seed = 2
rand = np.random.RandomState(seed=seed)

# Model parameters; no. of message-passing steps
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters
num_training_iterations = 100
batch_size_tr = 32
batch_size_ge = 100

# Input and target placeholders
input_ph, target_ph = create_placeholders(batch_size_tr)

# Connect the data to the model and instantiate
model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
# A list of outputs, one per processing step
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss
loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps
# QUESTION: What is this for?
# I believe this is because we want to minimize
# the training loss over all processing steps
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
# Test/generalization loss
loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step

# Optimizer
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

# Lets an iterable of TF graphs be output from a session as NP graphs
input_ph, target_ph = make_all_runnable_in_session(
    input_ph, target_ph)  # QUESTION: What is this?


##################
# TRAINING/TESTING
##################


try:
    sess.close()
except NameError:
    pass

sess = tf.Session()
sess.run(tf.global_variables_initializer())

last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.

# How much time between logging and printing the current results.
log_every_seconds = 20

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training loss), Lge (test/generalization loss), "
      "Ctr (training fraction nodes/edges labeled correctly), "
      "Str (training fraction examples solved correctly), "
      "Cge (test/generalization fraction nodes/edges labeled correctly), "
      "Sge (test/generalization fraction examples solved correctly)")

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    feed_dict, _ = create_feed_dict(batch_size_tr, input_ph, target_ph)
    train_values = sess.run({
        "step": step_op,
        "target": target_ph,
        "loss": loss_op_tr,
        "outputs": output_ops_tr
    }, feed_dict=feed_dict)
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time

    # print(train_values)
    # with open("../figures/pickles/train_vals{}.pkl".format(iteration), "wb") as train_val_file:
    #     pickle.dump(train_values, train_val_file)

    if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time
        feed_dict, raw_graphs = create_feed_dict(
            batch_size_ge, input_ph, target_ph)
        test_values = sess.run({
            "target": target_ph,
            "loss": loss_op_ge,
            "outputs": output_ops_ge
        },
            feed_dict=feed_dict)
        correct_tr, solved_tr = compute_accuracy(
            train_values["target"], train_values["outputs"][-1], use_edges=True)
        correct_ge, solved_ge = compute_accuracy(
            test_values["target"], test_values["outputs"][-1], use_edges=True)
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        corrects_tr.append(correct_tr)
        solveds_tr.append(solved_tr)
        losses_ge.append(test_values["loss"])
        corrects_ge.append(correct_ge)
        solveds_ge.append(solved_ge)
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
              " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
                  iteration, elapsed, train_values["loss"], test_values["loss"],
                  correct_tr, solved_tr, correct_ge, solved_ge))

# save model
saver = tf.train.Saver()
saver.save(sess, "./model.ckpt")

out_df = pd.DataFrame(np.array([logged_iterations, losses_tr, losses_ge,
    corrects_tr, solveds_tr, corrects_ge, solveds_ge]).T,
    columns=["iteration", "loss_tr", "loss_ge", "correct_tr", "solved_tr",
        "correct_ge", "solved_ge"])
out_df.to_pickle('so1ftmax_ce_loss.pkl')

# Plot results curves.
fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Loss.
y_tr = losses_tr
y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss (binary cross-entropy)")
ax.legend()
# Correct.
y_tr = corrects_tr
y_ge = corrects_ge
ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction correct across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction nodes/edges correct")
# Solved.
y_tr = solveds_tr
y_ge = solveds_ge
ax = fig.add_subplot(1, 3, 3)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction solved across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction examples solved")
plt.savefig("../figures/plots.png", dpi=1000)


# validation on fully trained model
batch_size_v = 1000
num_processing_steps_v = 10

input_ph_v, target_ph_v = create_placeholders(batch_size_v)

output_ops_v = model(input_ph_v, num_processing_steps_v)

loss_ops_v = create_loss_ops(target_ph_v, output_ops_v)
loss_op_v = loss_ops_v[-1]

input_ph_v, target_ph_v = make_all_runnable_in_session(input_ph_v, target_ph_v)

feed_dict_v, raw_graphs_v = create_feed_dict(batch_size_v, input_ph_v, target_ph_v)
validation_values = sess.run({
    "target": target_ph_v,
    "loss": loss_op_v,
    "outputs": output_ops_v
},
    feed_dict=feed_dict_v)
correct_v, solved_v = compute_accuracy(
    validation_values["target"], validation_values["outputs"][-1], use_edges=True)
loss_v = validation_values["loss"]

# save validation stuffs
pickle.dump(feed_dict_v, "../figures/pickles/feed_dict_v.pkl")
pickle.dump(raw_graphs_v, "../figures/pickles/raw_graphs_v.pkl")
pickle.dump(validation_values, "../figures/pickles/validation_values.pkl")

# validation statistics
print(loss_v)
print(correct_v)
print(solved_v)
