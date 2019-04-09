from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform

def download():
	categories=['tiger','kittycat']
	for category in categories:
		os.makedirs('./for_transfer_learning/data/%s' %catagory,exist_ok=True)
		with open('./for_transfer_learning/imagenet_%s.txt' %category,'r') as file:
			urls=file.readlines()
			n_urls=len(urls)
			for i,url in enumerate(urls):
				try:
					urlretrieve(url.strip(),'./for_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
					print('%s %i %i' %(category,i,n_urls))
				except:
					print('%s %i %i' %(category,i,n_urls),'no image')
					
def load_img(path):
	img=skimage.io.imread(path)
	img=img/255.0



class Vgg16:
	vgg_mean=[103.939,116.779,123.68]
	def __init__(self,vgg16_npy_path=None,restore_from=None):
		try:
			self.data_dict=np.load(vgg16_npy_path,encoding='latin1').item()
		except FileNotFoundError:
			print('please download VGG16 parameters from here')
		self.tfx=tf.placeholder(tf.float32,[None,224,224,3])
		self.tfx=tf.placeholder(tf.float32,[None,1])
		
		red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])
        conv1_1=self.conv_layer(bgr,'conv1_1')
        conv1_2=self.conv_layer(conv1_1,'conv1_2')
        pool1=self.max_pool(conv1_2,'pool1')
        
        conv2_1=self.conv_layer(pool1,'conv2_1')
        conv2_2=self.conv_layer(conv2_1,'conv2_2')
        pool2=self.max_pool(conv2_2,'pool2')
        
        conv3_1=self.conv_layer(pool2,'conv3_1')
        conv3_2=self.conv_layer(conv3_1,'conv3_2')
        conv3_3=self.conv_layer(conv3_2,'conv3_3')
        pool3=self.max_pool(conv3_3,'pool3')
        
        conv4_1=self.conv_layer(pool3,'conv4_1')
        conv4_2=self.conv_layer(conv4_1,'conv4_2')
        conv4_3=self.conv_layer(con4_2,'conv4_3')
        pool4=self.max_pool(conv4_3,'pool4')
        
        conv5_1=self.conv_layer(pool4,'conv5_1')
        conv5_2=self.conv_layer(conv5_1,'conv5_2')
        conv5_3=self.conv_layer(conv5_2,'conv5_3')
        pool5=self.max_pool(conv5_3,'pool5')
        
        self.flatten=tf.reshape(pool5,[-1,7*7*512])
        self.fc6=tf.layers.dense(self.flatten,256,tf.nn.relu,name='fc6')
        self.out=tf.layers.dense(self.fc6,1,name='out')
        
        self.sess=tf.Session()
        if restore_from:
			saver=tf.train.Saver()
			saver.restore(self.sess,restore_from)
		else:
			self.loss=tf.losses.mean_squared_error(labels=self.tfy,predictions=self.out)
			self.train_op=tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
			self.sess.run(tf.global_variables_initializer())
			
	def max_pool(self,bottom,name):
		return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides[1,2,2,1],padding='SAME',name=name)
		
	def conv_layer(self,bottom,name):
		with tf.variable_scope(name):
			conv=tf.nn.conv2d(bottom,self.data_dict[name][0],[1,1,1,1],padding='SAME')
			lout=tf.nn.relu(tf.nn.bias_add(conv,self.data_dict[name][1]))
			return lout
			
	def train(self,x,y):
		loss,_=self.sess.run([self.loss,self.train_op],{self.tfx:x,self.tfy:y})
		return loss
		
	def predict(self,paths):
		
def train():
	
        
        
		
