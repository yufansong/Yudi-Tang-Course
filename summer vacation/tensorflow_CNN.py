import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #stride[batch,x_movement,y_movement,channel]一般不对这两个参数做设置
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#第一个和最后一个元素都必须等于1

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784])#28*28
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32,None)
x_image=tf.reshape(xs,[-1,28,28,1])#-1代表先不管图片数量，电脑会自动计算导入了多少张图片,最后一个1是channel
#print(x_image,shape)


##conv1 layer##
W_conv1=weight_variable([5,5,1,32])#patch 5*5, in size 1,out size 32 ,32相当于是一个图像的厚度，
                                   #因为CNN是通过增加图像的channel来总结图片的特征，32是随便取的，多了少了只是会影响精度
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output size 28*28*32
h_pool1=max_pool_2x2(h_conv1)                       #output size 14*14*32


##conv2 layer##
W_conv2=weight_variable([5,5,32,64])#patch 5*5, in size 32,out size 64
                                   #因为CNN是通过增加图像的channel来总结图片的特征，32是随便取的，多了少了只是会影响精度
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output size 28*28*64
h_pool2=max_pool_2x2(h_conv2)                       #output size 7*7*64

##func1 layer ##
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,0.5)

##func2 layer##
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)



## the error between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                             reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()
#important step
sess.run(tf.initialize_all_variables())

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(
            mnist.test.images,mnist.test.labels
        ))