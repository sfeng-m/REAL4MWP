# coding=utf-8
"""Some tool in common."""

def isnumber(aString):
    try:
        float(aString)
        return True
    except:
        return False


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b