from singleton import Singleton
from resource_manager import ResourceManager
from random import randint
from itertools import product
from time import sleep
from rllite import SAC
import multiprocessing as mp

def run(lock, shared_eps_num, shared_eps_reward, hyper_param):    
    model = SAC(
        env_name = 'Pendulum-v0',
        load_dir = './ckpt/ckpt_'+str(hyper_param[0])+'_'+str(hyper_param[1]),
        log_dir = './log/log_'+str(hyper_param[0])+'_'+str(hyper_param[1]),
        buffer_size = 1e6,
        seed = hyper_param[1],
        max_episode_steps = 500, # manual set
        batch_size = hyper_param[0],
        discount = 0.99,
        learning_starts = 500,
        tau = 0.005,
        save_eps_num = 100
	)

    timesteps = 0
    total_timesteps = 1e5
    max_eps_steps = 100
    
    # train
    while timesteps < total_timesteps:
        episode_reward = 0
        done = False
        eps_steps = 0
        obs = model.env.reset()
        while not done and eps_steps < max_eps_steps:
            action = model.predict(obs)
            new_obs, reward, done, info = model.env.step(action)
            model.replay_buffer.push(obs, action, reward, new_obs, done)
            obs = new_obs
            episode_reward += reward
            eps_steps += 1
            timesteps += 1
            if timesteps > model.learning_starts :
                model.train_step()
        model.episode_num += 1
        model.writer.add_scalar('episode_reward', episode_reward, model.episode_num)
        
        lock.acquire()
        shared_eps_num.value = model.episode_num
        shared_eps_reward.value = episode_reward
        lock.release()
        
class ATR(Singleton):
    def __init__(self, hyper_params, max_num=9999, random=True):
        self.hp_name = list(hyper_params.keys())
        self.hp = self.get_hp(hyper_params)
        self.max_num = max_num
        self.random = random

        self.resource_manager = ResourceManager(mem_limit=1, cpu_limit = 0.1, gpu_limit=0.5, max_instances=self.max_num)

        self.waiting_pool = [item for item in self.hp]
        self.finished_pool = []
        self.working_pool = []
        self.working_process = []
        
        self.lock_list = []
        self.shared_eps_num_list = []
        self.shared_eps_reward_list = []
        
    def get_hp(self, hyper_params):
        hp_list = list(hyper_params.values())
        if len(hp_list) < 2:
            return hp_list

        hp = hp_list.pop(0)
        for i in range(len(hp_list)):
            hp = list(product(hp, hp_list[i]))
        return hp
        
    def start(self):
        while True:
            self.auto_tune()
            sleep(1.0)
    
    def report(self):
        self.resource_manager.report()
        
    def auto_tune(self):
        self.ask_result()
        self.auto_kill()
        self.auto_gen()
        
    # for test
    def ask_result(self):
        if len(self.working_pool) == 0: return
        for i in range(len(self.working_pool)):
            lock = self.lock_list[i]
            lock.acquire()
            shared_eps_num = self.shared_eps_num_list[i].value
            shared_eps_reward = self.shared_eps_reward_list[i].value
            lock.release()
            print(i, shared_eps_num, shared_eps_reward)
    
    # for test
    def auto_kill(self):
        if len(self.working_pool) == 0: 
            if len(self.waiting_pool) == 0:
                print('All Job Finished !')
            return

        index = -999
        min_eps_reward = -99999
        for i in range(len(self.working_pool)):
            lock = self.lock_list[i]
            lock.acquire()
            shared_eps_num = self.shared_eps_num_list[i].value
            shared_eps_reward = self.shared_eps_reward_list[i].value
            lock.release()
            if shared_eps_num > 1000:
                if shared_eps_reward < min_eps_reward:
                    min_eps_reward = shared_eps_reward
                    index = i
        if index < 0:
            return
        process = self.working_process.pop(index)
        process.terminate()
        hyper_param = self.working_pool.pop(index)
        self.finished_pool.append(hyper_param)
        self.lock_list.pop(index)
        self.shared_eps_num_list.pop(index)
        self.shared_eps_reward_list.pop(index)
        
    def auto_gen(self):
        while True:
            if len(self.waiting_pool) == 0: return
            if len(self.working_pool) >= self.max_num: return
            if not self.resource_manager.get_memory_access(): return
            if not self.resource_manager.get_cpu_access(): return
            gpu_id = self.resource_manager.get_gpu_access()
            if gpu_id < 0: return
            if(self.random):
                index = 0
                if len(self.waiting_pool) > 1:
                    index = randint(0, len(self.waiting_pool)-1)
                hyper_param = self.waiting_pool.pop(index)
                self.working_pool.append(hyper_param)
            else:
                hyper_param = self.waiting_pool.pop(0)
                self.working_pool.append(hyper_param)

            self.create_process(hyper_param)
            
    def create_process(self, hyper_param):
        ctx = mp.get_context('spawn')
        lock = ctx.Lock()
        shared_eps_num = ctx.Value('l', 0)
        shared_eps_reward = ctx.Value('d', 0.0)
        
        process = ctx.Process(target=run, args=(lock, shared_eps_num, shared_eps_reward, hyper_param))
        process.start()
        
        self.lock_list.append(lock)
        self.shared_eps_num_list.append(shared_eps_num)
        self.shared_eps_reward_list.append(shared_eps_reward)
        self.working_process.append(process)

        print('Start:', hyper_param)
            
    def listener(self, event):
        if event.exception: print('The job crashed :(')

if __name__ == '__main__':
    hyper_params = {
            'batch_size':[32, 64, 128],
            'seed':[1,2,3]
            }
    atr = ATR(hyper_params, max_num=4)
    atr.start()