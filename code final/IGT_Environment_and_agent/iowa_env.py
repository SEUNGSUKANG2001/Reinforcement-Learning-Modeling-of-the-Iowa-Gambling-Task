import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# 환경 클래스 정의
class Iowa_Gambling_Task(gym.Env):
    def __init__(self):
        super(Iowa_Gambling_Task, self).__init__() ## 부모 클래스 생성자 호출
        
        ## variable
        self.action_space = gym.spaces.Discrete(4) ## A,B,C,D 4가지 선택지
        self.score = 2000 ##사람이 가지고 있는 돈 ## 처음 시작하는 돈 
        self.observation_space = 0 ## 임의로 0으로 둠
        self.current_step = 0 ## 현재 step ## 첫 step = 0
        
        ## memeory
        self.choices = [] ## 지금까지 한 선택들
        self.rewards = [] ## 지금까지 받은 reward들

    def create_deck_rewards(self, action):
        ## reward
        positive_rewards ={ 
            0 : 100, #A
            1 : 100, #B
            2 : 50, #C
            3 : 50  #D   
        }
        
        negative_rewards={ ## lambda : 쓰는 이유: 선택할 때마다 바뀌게 하기 위해서
            0 : lambda : np.random.choice([0, 150, 200, 250, 300, 350], p =[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]),## A 
            1 : lambda : np.random.choice([0, 1250], p =[0.9, 0.1]),## B
            2 : lambda : np.random.choice([0, 25, 75, 50], p =[0.5, 0.1, 0.1, 0.3]), ## C
            3 : lambda : np.random.choice([0, 250], p =[0.9, 0.1]) ## D
        }

        return positive_rewards[action], negative_rewards[action]()
    
    def reset(self):
        self.rewards = []
        self.choices = []
        self.current_step = 0
        self.score = 2000 
        return 0, {}

    def step(self, action):
        ## reward 계산
        pos_r , neg_r = self.create_deck_rewards(action)
        reward = pos_r -  neg_r
        
        ## memory
        self.choices.append(action)
        self.rewards.append(reward)
        
        ## reward 반영
        self.score += reward
        self.current_step += 1 ## 현재 step 갱신

        ## 게임 끝났는 지 판단
        done = False 
        if self.score<=0:
            done = True

        ## info
        info = {
            "positive_reward" : pos_r,
            "negative_reward" : neg_r
        }
        
        return 0, reward, done, info

    def get_history(self): ## 선택과 reward return
        return self.choices, self.rewards

    def get_score(self): ## score return
        return self.score
    
    def render(self): ## render
        print(f"Step : {self.current_step}, Score : {self.score}")