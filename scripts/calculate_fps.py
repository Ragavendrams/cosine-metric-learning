import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Function to load frozen graph
def load_frozen_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name='')
        return g

def main():
    print('Calculating speed of baseline model')

    # Load input RGB image
    image_path = "/home/mlexplorer/Documents/Person_ReID/dataset/train_cuhk/1/0001C00T0000F2555.jpg"
    image = Image.open(image_path)
    image_np = np.asarray(image)
    image_np = np.expand_dims(image_np, axis=0)

    # Load frozen graph
    graph = load_frozen_pb('./cuhk.pb')
    sess = tf.Session(graph=graph)
    output = sess.graph.get_tensor_by_name('features:0')
    input = sess.graph.get_tensor_by_name('images:0')

    # Measure execution time
    num_runs = 1000
    t1 = 0.0
    for i in range(num_runs):
        t0 = time.time()
        features = sess.run(output, feed_dict={input:image_np})
        if i > 0:
            t1 += (time.time() - t0)
    print(f'Execution time: {t1/(num_runs-1)}, FPS: {(num_runs-1)/t1}')

def main_features():
    print('Calculating speed of improved model.. ')
    graph = load_frozen_pb('./cuhk_features.pb')
    sess = tf.Session(graph=graph)
    output = sess.graph.get_tensor_by_name('features:0')
    input = sess.graph.get_tensor_by_name('images:0')
    image_np = np.random.uniform(0.0, 6.0, size=(1,64,32,32))

    # Measure execution time
    num_runs = 1000
    t1 = 0.0
    for i in range(num_runs):
        t0 = time.time()
        features = sess.run(output, feed_dict={input:image_np})
        if i > 0:
            t1 += (time.time() - t0)
    print(f'Execution time: {t1/(num_runs-1)}, FPS: {(num_runs-1)/t1}')

if __name__ == "__main__":
    main()
    main_features()
