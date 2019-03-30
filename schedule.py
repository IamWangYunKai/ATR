import pynvml
import subprocess
from random import randint
from itertools import product
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)  
        return cls._instance  

class GPUManager(Singleton):
    # Minimal free memory(MiB) for new process to run
    def __init__(self, min_free_mem=0):
        pynvml.nvmlInit()
        self.min_free_mem = min_free_mem
        self.gpu_num = pynvml.nvmlDeviceGetCount()
        self.handles = []
        for gpu_id in range(self.gpu_num):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            print("Find GPU", gpu_id, ":", pynvml.nvmlDeviceGetName(handle).decode('utf-8'))
            self.handles.append(handle)
        
    def __del__(self):
        pynvml.nvmlShutdown()
        
    def get_access(self):
        free_mems = []
        for handle in self.handles:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # info.total,info.free,info.used (B)
            free_mems.append(info.free)
        max_free_mem = max(free_mems)
        gpu_id = free_mems.index(max_free_mem)
        if max_free_mem/1024/1024 < self.min_free_mem:
            # no enough memory to use
            return -1
        else:
            # return gpu id with maximal memory
            return gpu_id
    
class ATR(Singleton):
    def __init__(self, script, hyper_params, max_num=None, random=True):
        self.script = script
        self.hp_name = list(hyper_params.keys())
        self.hp = self.get_hp(hyper_params)
        self.max_num = max_num
        self.random = random
        
        self.scheduler = BackgroundScheduler()
        self.gpu_manager = GPUManager(600)
        
        # start_date='2019-03-30 18:29:00', end_date='2019-03-30 18:30:00'
        self.scheduler.add_job(self.auto_tune, args=(666,), trigger='interval', seconds =1, id='auto_tune')
        self.scheduler.add_listener(self.listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.waiting_pool = [item for item in self.hp]
        self.finished_pool = []
        self.working_pool = []
        self.working_process = []
        
    def get_hp(self, hyper_params):
        hp_list = list(hyper_params.values())
        if len(hp_list) < 2:
            return hp_list

        hp = hp_list.pop(0)
        for i in range(len(hp_list)):
            hp = list(product(hp, hp_list[i]))
        return hp
        
    def start(self):
        self.scheduler.start()    
    
    def auto_tune(self, arg):
        print('ask result', arg)
        self.ask_result()
        self.auto_kill()
        self.auto_gen()
        
    def ask_result(self):
        if len(self.working_ids) == 0: return
        pass
    
    def auto_kill(self):
        if len(self.working_ids) == 0: return
        pass
    
    def auto_gen(self):
        while True:
            if len(self.waiting_pool) == 0: return
            if len(self.working_pool) == self.max_num: return
            gpu_id = self.gpu_manager.get_access()
            if gpu_id < 0: return
            if(self.random):
                index = randint(0, len(self.waiting_pool)-1)
                hyper_param = self.waiting_pool.pop(index)
                self.working_pool.append(hyper_param)
            else:
                hyper_param = self.waiting_pool.pop(0)
                self.working_pool.append(hyper_param)
                
            cmd = 'python ' + self.script
            for i in range(len(self.hp_name)):
                if isinstance(hyper_param[i], str):
                    cmd += ' --' + self.hp_name[i] + ' ' + hyper_param[i] + ' '
                else:
                    cmd += ' --' + self.hp_name[i] + ' ' + str(hyper_param[i]) + ' '
            cmd += ' --cuda ' + str(gpu_id)
            process = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            #print(process.stdout.read())
            self.working_process.append(process)
            print(process.pid, cmd)
            break
            
    def listener(self, event):
        if event.exception: print('The job crashed :(')

if __name__ == '__main__':
    script = 'child.py'
    hyper_params = {
            'policy':['mlp', 'rnn', 'cnn'],
            'seed':[1,2,3,4,5]
            }
    atr = ATR(script, hyper_params, max_num=8)
    atr.auto_gen()