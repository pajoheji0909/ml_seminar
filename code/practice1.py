import os
import numpy as np
import tensorflow as tf

# to deactivate warnings (if you didn't installed TF by source)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess_example = tf.Session()
# string type constant
string_ex = tf.constant("Hello, it's String.")
string_out = sess_example.run(string_ex)

# initialization
init = tf.initialize_all_variables()
sess_example.run(init)

# check the types of constant and sess.run()
print("Type of 'string_ex': %s" % (type(string_ex)))
print(string_ex)
print("Type of 'string_out': %s" % (type(string_out)))
print(string_out)

# float32 type constant
float_ex1 = tf.constant(3.)
float_ex2 = tf.constant(2.5)

print("Type of 'float_ex': %s" % (type(float_ex1)))
print(float_ex1)
print("You can add constant : %s" % (type(float_ex1+float_ex2)))
print(float_ex1 + float_ex2)

# operator tf.add
print("tf.add()")
add_op = tf.add(float_ex1, float_ex2)
add_out = sess_example.run(add_op)
print(add_out)

# initialization
init = tf.initialize_all_variables()
sess_example.run(init)

# variable
weight = tf.Variable(tf.random_normal([20, 10], stddev=0.1))
bias = tf.Variable(tf.zeros([2, 10]))
print("weight : %s Type is %s" % (weight, type(weight)))
print("bias : %s Type is %s" % (bias, type(bias)))

# placeholder
ph_ex = np.random.rand(1, 20)
in_ph = tf.placeholder(tf.float32, [None, 20])
print(" in_ph : %s \n type is %s " % (in_ph, type(in_ph)))

# initialization
init = tf.initialize_all_variables()
sess_example.run(init)

# simple function
oper = tf.matmul(in_ph, weight) + bias
val = sess_example.run(oper, feed_dict={in_ph: ph_ex})
print("oper is %s \n type is %s" % (oper, type(oper)))
print("val is %s \n type is %s" % (val, type(val)))
