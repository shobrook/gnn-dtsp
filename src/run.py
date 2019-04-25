#########
# GLOBALS
#########


import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle
from graph_nets import utils_np
from graph_nets.demos import models

# Local modules
import utils


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

    cs, ss = [], []
    for td, od in zip(tdds, odds):

        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)

        c = [xe == ye] if use_edges else []
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
input_ph, target_ph = utils.create_placeholders(batch_size_tr)

# Connect the data to the model and instantiate
model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
# A list of outputs, one per processing step
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss
loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
# Test/generalization loss
loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step

# Optimizer
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

# Lets an iterable of TF graphs be output from a session as NP graphs
input_ph, target_ph = utils.make_all_runnable_in_session(input_ph, target_ph)


##########
# TRAINING
##########


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

    feed_dict, _ = utils.create_feed_dict(batch_size_tr, input_ph, target_ph)
    train_values = sess.run({
        "step": step_op,
        "target": target_ph,
        "loss": loss_op_tr,
        "outputs": output_ops_tr
    }, feed_dict=feed_dict)

    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time

    if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time

        feed_dict, raw_graphs = utils.create_feed_dict(
            batch_size_ge, input_ph, target_ph)
        test_values = sess.run({
            "target": target_ph,
            "loss": loss_op_ge,
            "outputs": output_ops_ge
        }, feed_dict=feed_dict)

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


#########
# TESTING
#########


test_batch_size = 1000
num_processing_steps_test = 10

test_input_ph, test_target_ph = utils.create_placeholders(test_batch_size)
test_output_ops = model(test_input_ph, num_processing_steps_test)

test_loss_ops = create_loss_ops(test_target_ph, test_output_ops)
test_loss_op = test_loss_ops[-1]

test_input_ph, test_target_ph = utils.make_all_runnable_in_session(
    test_input_ph, test_target_ph)

test_feed_dict, test_input_graphs = utils.create_feed_dict(
    test_batch_size, test_input_ph, test_target_ph)
test_values = sess.run({
    "target": test_target_ph,
    "loss": test_loss_op,
    "outputs": test_output_ops
}, feed_dict=test_feed_dict)


#########
# STORAGE
#########


# Store the model
saver = tf.train.Saver()
saver.save(sess, "../data/pickles/model.ckpt")

# Store the training statistics
train_stats = pd.DataFrame(np.array([logged_iterations, losses_tr, losses_ge,
                                     corrects_tr, solveds_tr, corrects_ge, solveds_ge]).T,
                           columns=["iteration", "loss_tr", "loss_ge", "correct_tr", "solved_tr",
                                    "correct_ge", "solved_ge"])
train_stats.to_pickle("../data/pickles/train_stats.pkl")

# Store the test results
pickle.dump({
    "outputs": utils_np.graphs_tuple_to_networkxs(test_values["outputs"][-1]),
    "targets": utils_np.graphs_tuple_to_networkxs(test_values["target"]),
    "inputs": test_input_graphs,
}, open("../data/pickles/test_results.pkl", "wb"))
