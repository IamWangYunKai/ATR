import subprocess
import pynvml

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
        
if __name__ == '__main__':
    print('analyze start....' )
    gpu_manager = GPUManager(600)
    gpu_id = gpu_manager.get_access()

    policys = ['mlp', 'rnn', 'cnn']
    seeds = [1,2,3]
    
    for policy in policys:
        for seed in seeds:
            p=subprocess.Popen(
                    'python child.py' + \
                    ' --policy '+policy + \
                    ' --seed '+ str(seed) + \
                    ' --cuda '+ str(gpu_id),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell = True
                    )
            print(p.pid)
            print(p.stdout.readlines())
            #p.kill()
                
    print('analyze finised!!!')