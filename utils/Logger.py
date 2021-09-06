import os
import logging

def initlog(logfile):
    path = './logs'
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger()  # 实例化一个logger对象
    logger.setLevel(logging.INFO)  # 设置初始显示级别
    if (len(logger.handlers) == 0):  # 只创建一个句柄，当句柄存在时不创建，防止同个进程多次调用initlog()时相同日志信息重复写入；
        # 创建一个文件句柄
        file_handle = logging.FileHandler(logfile, encoding="UTF-8")
        # 创建一个流句柄
        stream_handle = logging.StreamHandler()
        # 创建一个输出格式
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                                datefmt="%a, %d %b %Y %H:%M:%S")
        file_handle.setFormatter(fmt)  # 文件句柄设置格式
        stream_handle.setFormatter(fmt)  # 流句柄设置格式
        logger.addHandler(file_handle)  # logger对象绑定文件句柄
        logger.addHandler(stream_handle)  # logger对象绑定流句柄
    return logger