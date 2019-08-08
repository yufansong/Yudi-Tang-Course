from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
import tensorflow as tf

digits=datasets.load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=3)


def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])

#add output layer
l1=add_layer(xs,64,100,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)

#the loss between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                             reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
mergerd=tf.summary.merge_all()
#summary writer goes in here
train_writer=tf.summary.FileWriter('logs/train',sess.graph)
test_writer=tf.summary.FileWriter('logs/test',sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
    if i%50==0:
        train_result=sess.run(mergerd,feed_dict={xs:X_train,ys:y_train})
        test_result = sess.run(mergerd, feed_dict={xs: X_train, ys: y_train})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)