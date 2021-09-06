# -*- encoding: UTF-8 -*-
"""
@author: 'Shifeng Huang'
@Company：CVTE-RESEARCH
@time: '2021.3.15'
@describe: 等式归一化的相关代码；
"""

import random 
import copy

equation_list = ['x', '=', 'temp_a', '-', '(', '-', 'temp_b', ')']
def norm_equ(equ_list):
    '''
    实现加法和乘法的归一化（即a+d+c+b归一化为a+b+c+d），最长实现4个数字连加/乘，可添加
    '''
    i = 0
    new_equ_list = []
    while i < len(equ_list):
        used_flag = False
        if 'temp' in equ_list[i] and (i+6) < len(equ_list) and equ_list[i+1] == '+' and 'temp' in equ_list[i+2] and equ_list[i+3] == '+' and 'temp' in equ_list[i+4] and equ_list[i+5] == '+' and 'temp' in equ_list[i+6]:  
            if not ((i-1>=0 and equ_list[i-1] in ['/','-', '*', '^']) or (i+7<len(equ_list) and equ_list[i+7] in ['/','*', '^'])):
                temp = [equ_list[i], equ_list[i+2], equ_list[i+4], equ_list[i+6]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]+['+']+sort_temp[2:3]+['+']+sort_temp[3:4]
                new_equ_list += new_temp
                i += 7
                used_flag = True
        if not used_flag and 'temp' in equ_list[i] and (i+6) < len(equ_list) and equ_list[i+1] == '*' and 'temp' in equ_list[i+2] and equ_list[i+3] == '*' and 'temp' in equ_list[i+4] and equ_list[i+5] == '*' and 'temp' in equ_list[i+6]:  
            if not ((i-1>=0 and equ_list[i-1] in ['/', '^']) or (i+7< len(equ_list) and equ_list[i+7] in ['/', '^'])):
                temp = [equ_list[i], equ_list[i+2], equ_list[i+4], equ_list[i+6]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3]+['*']+sort_temp[3:4]
                new_equ_list += new_temp
                i += 7
                used_flag = True
        if not used_flag and 'temp' in equ_list[i] and (i+4) < len(equ_list) and 'temp' in equ_list[i+2] and equ_list[i+1] == '+' and equ_list[i+3] == '+' and 'temp' in equ_list[i+4]:  
            if not ((i-1>=0 and equ_list[i-1] in ['/','-', '*', '^']) or (i+5<len(equ_list) and equ_list[i+5] in ['/','*', '^'])):
                temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]+['+']+sort_temp[2:3]
                new_equ_list += new_temp
                i += 5
                used_flag = True
        if not used_flag and 'temp' in equ_list[i] and (i+4) < len(equ_list) and 'temp' in equ_list[i+2] and equ_list[i+1] == '*' and equ_list[i+3] == '*' and 'temp' in equ_list[i+4]:  
            if not ((i-1>=0 and equ_list[i-1] in ['/', '^']) or (i+5< len(equ_list) and equ_list[i+5] in ['/', '^'])):
                temp = [equ_list[i], equ_list[i+2], equ_list[i+4]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]+['*']+sort_temp[2:3]
                new_equ_list += new_temp
                i += 5
                used_flag = True
        if not used_flag and 'temp' in equ_list[i] and (i+2) < len(equ_list) and 'temp' in equ_list[i+2]  and equ_list[i+1] == '+' and 'temp' in equ_list[i+2] :            
            if not ((i-1>=0 and equ_list[i-1] in ['/','-', '*', '^']) or (i+3<len(equ_list) and equ_list[i+3] in ['/', '*', '^'])): 
                temp = [equ_list[i], equ_list[i+2]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['+']+sort_temp[1:2]
                new_equ_list += new_temp
                i += 3
                used_flag = True
        if not used_flag and 'temp' in equ_list[i] and (i+2) < len(equ_list) and 'temp' in equ_list[i+2] and equ_list[i+1] == '*' and 'temp' in equ_list[i+2] :
            if not ((i-1>=0 and equ_list[i-1] in ['/', '^']) or (i+3<len(equ_list) and equ_list[i+3] in ['/', '^'])):
                temp = [equ_list[i], equ_list[i+2]]
                sort_temp = sorted(temp)
                new_temp = sort_temp[0:1]+['*']+sort_temp[1:2]
                new_equ_list += new_temp
                i += 3
                used_flag = True
        if not used_flag:
            new_equ_list.append(equ_list[i])
            i+=1
    return new_equ_list[:]


def filter_repeat_num(equ_list):
    """过滤表达式中重复的数字，如a+b+c-b简化为a+c"""
    i = 0
    new_equ_list = []
    del_index = []
    while i < len(equ_list):
        if i not in del_index:
            if 'temp' in equ_list[i] and (i-1) >= 0 and equ_list[i-1] in ['+', '-'] and (i+1) < len(equ_list) and equ_list[i+1] not in ['*', '/'] and equ_list.count(equ_list[i]) > 1:
                unique_flag = True
                for j in range(i+1, len(equ_list)):
                    if equ_list[i] == equ_list[j] and j not in del_index and '(' not in equ_list[i:j] and ')' not in equ_list[i:j]:
                        if (equ_list[i-1] == '+' and equ_list[j-1] == '-') or (equ_list[i-1] == '-' and equ_list[j-1] == '+'):
                            if ((j+1) == len(equ_list)) or ((j+1) < len(equ_list) and equ_list[j+1] not in ['*', '/']):
                                del new_equ_list[-1]
                                del_index.append(j)
                                unique_flag = False
                                break
                if unique_flag:
                    new_equ_list.append(equ_list[i])
            elif i in del_index:
                del new_equ_list[-1]
            else:
                new_equ_list.append(equ_list[i])
        else:
            del new_equ_list[-1]
        i += 1
    return new_equ_list
    

if __name__ == '__main__':
    # equ_list = ['x', '=', 'temp_d', '*', 'temp_b', '*', 'temp_e', '*', 'temp_c']    
    # equation = norm_equ(equ_list)
    # equ_list = ['x', '=', 'temp_a', '+', 'temp_b', '+', 'temp_e', '-', 'temp_b'] 
    # equ_list = ['x', '=', 'temp_a', '+', 'temp_b', '*', 'temp_e', '-', 'temp_b'] 
    equ_list = ['x', '=', 'temp_d', '*', '(', 'temp_a', '+', 'temp_b', ')', '+', 'temp_e', '-', 'temp_b', '-', 'temp_e', '-', 'temp_e']    
    equation = filter_repeat_num(equ_list)
    print(equation)