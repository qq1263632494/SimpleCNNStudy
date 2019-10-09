# -*- coding: UTF-8 -*-
import sys, time


class ShowProcess:
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  # 进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        self.start = time.time()
        self.last_time = time.time()
        print('\033[1;36m开始训练\033[0m')

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  # 计算完成进度，格式为xx.xx%
        time_predict = (time.time() - self.last_time) * (self.max_steps - self.i)
        time_message = '，预计还有：' + '%.5f' % time_predict + '秒'
        time_message = '\033[1;33m' + time_message + '\033[0m'
        process_bar = '\033[1;35m[\033[0m' + '\033[1;31m>\033[0m' * num_arrow + ' ' * num_line + '\033[1;35m]\033[0m' \
                      + '\033[1;33m%.2f\033[0m' % percent + '\033[1;33m%\033[0m' + time_message + '\r'  # 带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar)  # 这两句打印字符到终端
        sys.stdout.flush()
        self.last_time = time.time()

    def close(self, words='训练完成,共计：'):
        print('')
        print('\033[1;36m' + words + str(time.time() - self.start) + ' 秒.' + '\033[0m')
        self.i = 0
