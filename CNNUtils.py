def cal_conv2d(x=0, padding=0, dialation=1, kernel_size=0, stride=1):
    import math
    return math.floor((x + 2 * padding - dialation * (kernel_size - 1) - 1) / stride + 1)


def cal_maxpool2d(x=0, padding=0, dialation=1, kernel_size=0, stride=1):
    stride = kernel_size
    import math
    return math.floor((x + 2 * padding - dialation * (kernel_size - 1) - 1) / stride + 1)


def cal_avgpool2d(x=0, padding=0, kernel_size=0, stride=1):
    stride = kernel_size
    import math
    return math.floor((x + 2 * padding - kernel_size) / stride + 1)
