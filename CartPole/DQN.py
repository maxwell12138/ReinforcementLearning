import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class DQN():
	def __init__(self,env,learning_rate=0.1,reward_decay=0.9,\
				e_greedy=0.3,replace_target_iter=4,memory_size=1000,batch_size=32):
		self.n_output=env.action_space.n
		self.n_input=env.observation_space.shape[0]
		self.alpha=learning_rate
		self.gamma=reward_decay
		self.epsilon=e_greedy
		self.replace_target_iter=replace_target_iter
		self.memory_size=memory_size
		self.batch_size=batch_size

		self.learn_step_counter=0
		self.memory=np.zeros((self.memory_size,self.n_input*2+3))
		self.costs=[]

		self._build_net()

		target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
		self.target_replace_op = [tf.assign(t,e) for t,e in zip(target_params,eval_params)]

		self.session=tf.Session()
		self.session.run(tf.initialize_all_variables())

	def _build_net(self):
		self.s=tf.placeholder(tf.float32,[None,self.n_input],name='s')
		self.s_=tf.placeholder(tf.float32,[None,self.n_input],name='s_')
		self.q_target=tf.placeholder(tf.float32,[None,self.n_output],name='q_target')

		target_net_name=['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
		eval_net_name=['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]

		n_hiddenlayer=5
		w_initializer=tf.random_normal_initializer(0,0.3)
		b_initializer=tf.constant_initializer(0.1)

		with tf.variable_scope('eval_net'):
			with tf.variable_scope('layer1'):
				w1=tf.get_variable('w1',[self.n_input,n_hiddenlayer],initializer=w_initializer,collections=eval_net_name)
				b1=tf.get_variable('b1',[1,n_hiddenlayer],initializer=b_initializer,collections=eval_net_name)
				l1=tf.nn.relu(tf.matmul(self.s,w1)+b1)
			with tf.variable_scope('layer2'):
				w2=tf.get_variable('w2',[n_hiddenlayer,self.n_output],initializer=w_initializer,collections=eval_net_name)
				b2=tf.get_variable('b2',[1,self.n_output],initializer=b_initializer,collections=eval_net_name)
				self.q_eval=tf.matmul(l1,w2)+b2

		with tf.variable_scope('target_net'):
			with tf.variable_scope('layer1'):
				w1=tf.get_variable('w1',[self.n_input,n_hiddenlayer],initializer=w_initializer,collections=target_net_name)
				b1=tf.get_variable('b1',[1,n_hiddenlayer],initializer=b_initializer,collections=target_net_name)
				l1=tf.nn.relu(tf.matmul(self.s_,w1)+b1)
			with tf.variable_scope('layer2'):
				w2=tf.get_variable('w2',[n_hiddenlayer,self.n_output],initializer=w_initializer,collections=target_net_name)
				b2=tf.get_variable('b2',[1,self.n_output],initializer=b_initializer,collections=target_net_name)
				self.q_next=tf.matmul(l1,w2)+b2

		with tf.name_scope('loss'):
			self.loss=tf.reduce_sum(tf.squared_difference(self.q_target,self.q_eval))
		with tf.name_scope('train'):
			self._train_op=tf.train.AdamOptimizer(self.alpha).minimize(self.loss)


	def store_memory(self,s,a,r,done,s_):
		if not hasattr(self,'memory_counter'):
			self.memory_counter=0
		index=self.memory_counter%self.memory_size
		self.memory[index,:]=np.hstack([s,a,r,done,s_])
		self.memory_counter+=1


	def get_memory_num(self):
		return self.memory_counter

	def choose_action(self,observation):
		q_eval=self.session.run(self.q_eval,feed_dict={self.s:[observation]})[0]
		if np.random.uniform()<self.epsilon:
			return np.random.randint(0,self.n_output-1)
		else:
			return np.argmax(q_eval)

	def action(self,observation):
		q_eval=self.session.run(self.q_eval,feed_dict={self.s:[observation]})[0]
		return np.argmax(q_eval)

	def learn(self):

		#update the network of q_target
		if self.learn_step_counter%self.replace_target_iter==0:
			self.session.run(self.target_replace_op)

		#select a batch of memories from pool
		if self.memory_counter>self.memory_size:
			container_size=self.memory_size
		else:
			container_size=self.memory_counter
		sample_index=np.random.choice(container_size,size=self.batch_size)
		batch_memory=self.memory[sample_index,:]

		# s,a,r,done,s_ in each piece of memory
		#extract s and s_ from memory, they are both 1*n_input vector,and calculate the corresponding q values
		q_next,q_eval=self.session.run([self.q_next,self.q_eval],\
									feed_dict={self.s:batch_memory[:, :self.n_input],\
												self.s_:batch_memory[:, -self.n_input:]})

		#get actions and rewards
		action_index=batch_memory[:,self.n_input].astype(int)
		reward=batch_memory[:,self.n_input+1]
		done=batch_memory[:,self.n_input+2]

		#find Qmax for next state
		q_next_max=np.max(q_next,axis=1)

		#Qtarget=reward+gamma*Qmax
		#based on q_eval, only few values of q_target are updated
		q_target=q_eval.copy()

		for i in range(self.batch_size):
			if done[i]:
				q_target[i,action_index[i]]=reward[i]
			else:
				q_target[i,action_index[i]]=reward[i]+self.gamma*q_next_max[i]

		_,cost=self.session.run([self._train_op,self.loss],feed_dict={self.s:batch_memory[:,:self.n_input],\
																	self.q_target:q_target})
		self.costs.append(cost)
		self.learn_step_counter+=1

	def save_data(self):
		fd=input('input file loction:')
		tf.train.Saver().save(self.session,fd+'\DQN.ckpt')

	def plot_loss(self):
		plt.plot(np.arange(len(self.costs)), self.costs)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()
