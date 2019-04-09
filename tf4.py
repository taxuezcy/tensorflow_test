import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE=64
LR_G=0.0001
LR_D=0.0001
N_IDEAS=5
ART_COMPONENTS=15
PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():
	a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
	paintings=a*np.power(PAINT_POINTS,2)+(a-1)
	return paintings
	
G_in=tf.placeholder(tf.float32,[None,N_IDEAS])
G_l1=tf.layers.dense(G_in,128,tf.nn.relu)
G_out=tf.layers.dense(G_l1,ART_COMPONENTS)

real_art=tf.placeholder(tf.float32,[None,ART_COMPONENTS],name='real_in')
D_10=tf.layers.dense(real_art,128,tf.nn.relu,name='1')
prob_artist0=tf.layers.dense(D_10,1,tf.nn.sigmoid,name='out')
D_11=tf.layers.dense(G_out,128,tf.nn.relu,name='1',reuse=True)
prob_artist1=tf.layers.dense(D_11,1,tf.nn.sigmoid,name='out',reuse=True)

D_loss=-tf.reduce_mean(tf.log(prob_artist0)+tf.log(prob_artist1))
G_loss=tf.reduce_mean(tf.log(1-prob_artist1))
train_D=tf.train.AdamOptimizer(LR_D).minimize(D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator'))
train_G=tf.train.AdamOptimizer(LR_G).minimize(G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator'))

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5000):
	artist_paintings=artist_works()
	G_ideas=np.random.randn(BATCH_SIZE,N_IDEAS)
	G_paintings,pa0,D1=sess.run([G_out,prob_artist0,D_loss,train_D,train_G],{G_in:G_ideas,real_art:artist_paintings})[:3]
	
	if step % 50==0:
	
