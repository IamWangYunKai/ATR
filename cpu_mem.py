# -*- coding: utf-8 -*-
import psutil
import os

info = psutil.virtual_memory()
print('This process memory used:', psutil.Process(os.getpid()).memory_info().rss/1024/1024)
print('Memory used:', info.used/1024/1024)
print('Memory free:',info.available/1024/1024)
print('Memory total:',info.total/1024/1024)
print('Memory percent:',info.percent, '%')
print('CPU num:',psutil.cpu_count())
print('CPU used:',psutil.cpu_percent(interval=1), '%')