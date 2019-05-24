"""
shared memory
"""
from time import sleep
from rllite import SAC
from multiprocessing import Process, Value, Lock

def run(lock, shared_eps_num, shared_eps_reward):    
    model = SAC(
    env_name = 'Pendulum-v0',
    load_dir = './ckpt',
    log_dir = "./log",
    buffer_size = 1e6,
    seed = 1,
    max_episode_steps = 500, # manual set
    batch_size = 64,
    discount = 0.99,
    learning_starts = 100,
    tau = 0.005,
    save_eps_num = 100
	)

    timesteps = 0
    total_timesteps = 1e6
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

if __name__ == '__main__':
    lock = Lock()
    shared_eps_num = Value('l', 0)
    shared_eps_reward = Value('d', 0.0)

    p = Process(target=run, args=(lock, shared_eps_num, shared_eps_reward))
    p.start()
    while True:
        sleep(1.0)
        lock.acquire()
        print(shared_eps_num.value, shared_eps_reward.value)
        lock.release()