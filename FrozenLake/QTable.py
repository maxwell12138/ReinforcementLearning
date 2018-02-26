import numpy as np

class QLearning():
	def __init__(self,n_actions,learning_rate=0.1,reward_decay=0.9,e_greedy=0.5):
		self.n_actions=n_actions
		self.alpha=learning_rate
		self.gamma=reward_decay
		self.epsilon=e_greedy
		self.q_table={}

	def egreedy_action(self,state):
		self.check_state_exist(state)
		if np.random.uniform()<self.epsilon:
			return np.random.randint(0,self.n_actions-1)
		else:
			return self.action(state)


	def action(self,state):
		q_values=self.q_table[state]
		action_indexes=np.argwhere(q_values==max(q_values))
		return np.random.choice(action_indexes[:,1])

	def learn(self,state,action,reward,next_state):
		self.check_state_exist(state)
		self.check_state_exist(next_state)
		q_next=self.q_table[next_state]
		q_next_max=max(q_next[0])
		q_state=self.q_table[state]
		q_target=reward+self.gamma*q_next_max
		q_state[0][action]+=self.alpha*(q_target-q_state[0][action])
		self.q_table.update({state:q_state})


	def check_state_exist(self, state):
		if not self.q_table.__contains__(state):
			self.q_table.setdefault(state,np.zeros([1,self.n_actions]))


	def printQtable(self):
		for key,value in self.q_table.items():
			print(key,':',value[0])
