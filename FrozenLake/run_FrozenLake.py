from QTable import QLearning
import gym


MAX_EPISODE=6000
ENV_NAME='FrozenLake-v0'
E_DECAY_TIME=20

if __name__=='__main__':
	env=gym.make(ENV_NAME)
	agent=QLearning(4)
	for n in range(MAX_EPISODE):
		if(n%5000==0):
			print(n)
		if n%(MAX_EPISODE//E_DECAY_TIME)==0:
			agent.epsilon=agent.epsilon*0.9
			#print('epsilon: ',agent.epsilon)
		#print('episode:',n)
		s=env.reset()
		while True:
			#env.render()
			action=agent.egreedy_action(s)
			s_,reward,done,info=env.step(action)
			agent.learn(s,action,reward,s_)
			s=s_
			if done:
				break

	agent.printQtable()
	print('train end')


	episode=1000
	win=0
	for i in range(episode):
		s=env.reset()
		while True:
			#env.render('human')
			action=agent.action(s)
			s_,reward,done,info=env.step(action)
			s=s_
			if done:
				if reward>0:
					win+=1
				break
	print('episode: ',episode)
	print('winning percentage:',win*100/episode,'%')
