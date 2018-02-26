from DQN import DQN
import gym
MAX_EPISODE=2000
STEP=300
ENV_NAME='CartPole-v0'

def update():
	env=gym.make(ENV_NAME)
	agent=DQN(env)

	for episode in range(MAX_EPISODE):
		s=env.reset()
		print('episode: ',episode)
		while True:
			action=agent.choose_action(s)
			s_,reward,done,_=env.step(action)

			agent.store_memory(s,action,reward,done,s_)
			if(agent.get_memory_num()>200):
				agent.learn()
			s=s_
			if done:
				break


	state=env.reset()
	num=1
	#total_reward=0
	while True:
		env.render()
		action=agent.action(state)
		state,reward,done,info=env.step(action)
		#total_reward+=reward
		if done:
			state=env.reset()
			print('num:',num)
			num+=1
		#elif total_reward>200:
		#	state=env.reset()


if __name__=='__main__':
	update()


