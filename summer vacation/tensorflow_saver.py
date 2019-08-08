import tensorflow as tf
import numpy as np

'''
## save to file
#remeber to define the same dtypr and shape when store
W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weight')
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init=tf.initialize_all_variables()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,'my_net/save_net')
    print('Save to path:',save_path)
'''
#restore varibel
#redefine the same shape and sanme type for your variables

W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weight')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

#not need init step

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_net')
    print('weights:',sess.run(W))
    print('biases:', sess.run(b))
