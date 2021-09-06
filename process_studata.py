# -*- encoding: UTF-8 -*-
"""
@author: 'stefan'
@describe: 数据预处理；
"""

import pandas as pd
import numpy as np
import random 
import re
import json
import jieba
import jieba.posseg as peg
from itertools import chain
from copy import deepcopy
from collections import defaultdict
import collections
import pickle
import os
import io
import sys
import math
from sklearn import metrics
from sympy import Integer, simplify
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
sys.path.append('..')
from config import arg_config
from preprocess.equation_norm import norm_equ, filter_repeat_num
from utils.tool import isnumber, is_equal


class PREFIX:

    def from_infix_to_prefix(self, expression):
        """将中序表达式转为先序，输入为list形式"""
        st = list()
        res = list()
        priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
        expression = deepcopy(expression)
        expression.reverse()
        for e in expression:
            if e in [")", "]"]:
                st.append(e)
            elif e == "(":
                c = st.pop()
                while c != ")":
                    res.append(c)
                    c = st.pop()
            elif e == "[":
                c = st.pop()
                while c != "]":
                    res.append(c)
                    c = st.pop()
            elif e in priority:
                while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                    res.append(st.pop())
                st.append(e)
            else:
                res.append(e)
        while len(st) > 0:
            res.append(st.pop())
        res.reverse()
        return res

    def expression2prefix(self, expression, answer, data_d=None, single_char=True, dataset='math23k'):
        """使用from_infix_to_prefix将中序表达式转成先序形式，并在数字后面加上'&'字符加以区分"""
        if dataset != 'mawps':
            expression = re.sub(r'(?<=\()(?P<op>-|\+)(?=\()', lambda x: '0'+x.group('op'), expression)  # 1-(-(1/2)) -> 1-(0-(1/2))
            expression = re.sub(r'(?P<value>\d+\.?\d*%)', lambda x: str(float(x.group('value')[:-1])/100), expression) # replace num% to num/100
            expression = re.findall(r'(?<!\)|\d)[-+]?[0-9]+\.[0-9]+|(?<!\)|\d)[-+]?[0-9]+|\^|\*|\/|\+|\-|\(|\)', expression)
            expression = self.from_infix_to_prefix(expression)
            exp_answer = self.compute_prefix_expression(expression)
        else:
            expression = self.from_infix_to_prefix(expression)
            digit_num_dict = eval(data_d['ques_digit_number'])
            num_expression = [digit_num_dict[tok] if tok in digit_num_dict else tok for tok in expression]
            try:
                exp_answer = self.compute_prefix_expression(num_expression)
            except:
                print('num_expression:{}, data:{}'.format(num_expression, data_d['original_text']))
                return None
        answer = eval(answer[:-1]) / 100 if '%' in answer else eval(answer)
        
        if exp_answer is not None and is_equal(exp_answer, answer):
            if single_char:
                new_equation = ''
                for eq in expression:
                    try:
                        ef = eval(eq)
                        eq += '&'
                        new_equation += eq
                    except:
                        new_equation += eq
                return new_equation
            else:
                return expression
        else:
            return None


    def compute_prefix_expression(self, pre_fix):
        """计算先序表达式的答案，输入必须是list,每个元素为切分好的字符"""
        st = list()
        operators = ["+", "-", "^", "*", "/"]
        pre_fix = deepcopy(pre_fix)
        pre_fix.reverse()
        for p in pre_fix:
            p = str(p)
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                    st.append(float(p[:-1]) / 100)
                else:
                    try:
                        st.append(eval(p))
                    except:
                        st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a + b)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a * b)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a * b)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if b == 0:
                    return None
                st.append(a / b)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a - b)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                # if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                if float(b) != 2.0 and float(b) != 3.0:
                    return None
                st.append(a ** b)
            else:
                return None
        if len(st) == 1:
            return st.pop()
        return None

    def split_equation(self, equation):
        """将表达式字符串分割成独立字符
        （该函数只在先序表达式中使用，先序表达式中为了区分不同数字，在数字的末尾加了'&'字符做区分，此处根据该符号切割字符串）
        """
        new_equation = []
        i = 0
        while i < len(equation):
            eq = equation[i]
            try:
                eq = int(eq)
                es = str(eq)
                while eq != '&' and i < len(equation)-1:
                    i += 1
                    eq = equation[i]
                    if eq != '&':
                        es += eq
                new_equation.append(es)
            except:
                new_equation.append(str(eq))
            i += 1
        return new_equation

    def prefix_to_postfix(self, equation):
        """将先序表达式转为中序表达式"""
        post_equation = list()
        for eq in equation[::-1]:
            if eq in ['+', '-', '*', '/', '^']:
                a = post_equation.pop(-1)
                b = post_equation.pop(-1)
                c = '({}{}{})'.format(a, eq, b)
                post_equation.append(c)
            else:
                post_equation.append(eq)
        # print(post_equation[0])
        # print(simplify(post_equation[0]))
        return post_equation[0]

    def merge_decimal(self, pred):
        """合并小数"""
        r_list = []
        i = 0
        while i < len(pred):
            if pred[i] == '.' and len(r_list) > 0:
                r_list[-1] = str(r_list[-1]) + str(pred[i]) + str(pred[i+1])
                i += 2
            else:
                r_list.append(pred[i])
                i += 1
        return r_list

    def split_decimal(self, pre_equation):
        """分割小数"""
        equation = []
        for eq in pre_equation:
            if isnumber(eq) and '.' in eq:
                if float(eq).is_integer():
                    equation.append(eq.split('.')[0])
                else:
                    equation.extend([eq[:eq.index('.')], '.', eq[eq.index('.')+1:]])
            else:
                equation.append(eq)
        return equation


class SplitQuestion:
    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
        else:
            return False
                
    def is_number(self, uchar):
        """判断一个unicode是否是数字"""
        if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
        else:
            return False
                
    def partition(self, text):
        """将未切分的question进行切分，同时保留数字连一起"""
        process_text = []
        text_len = len(text)
        for i, char in enumerate(text):
            if i == (text_len-1):
                process_text.append(char)
                continue
                
            if not self.is_number(char) and not self.is_alphabet(char) and char not in ['.', '%']:
                process_text.append(char+' ')
            elif self.is_alphabet(char):
                next_char = text[i+1]
                if self.is_alphabet(next_char):
                    process_text.append(char)
                else:
                    process_text.append(char+' ')
            else:
                next_char = text[i+1]
                if self.is_number(next_char) or next_char in ['.', '%']:
                    process_text.append(char)
                    continue
                else:
                    process_text.append(char+' ')
        question = ''.join(process_text).strip()
        question = re.sub('(\d+)/(\d+)', '\\1 / \\2', question)  # 1/5 -> 1 / 5
        question = re.sub('(\d+)\.(\d+)', '\\1 . \\2', question)  # 1.5 -> 1 . 5 [通过小数点进行切分]
        
        return question

class fetch_math23k(SplitQuestion):
    """
    处理math23k的数据集
    """
    def read_data_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_input(self, data_d, data, mode='train', single_char=True, dataset='math23k', split_number=True, norm_count=0, used_equ_norm=False):
        Prefix = PREFIX()
        if "千米/小时" in data_d["equation"]:
            data_d["equation"] = data_d["equation"][:-5]
        if data_d["equation"][:2] == 'x=' or data_d["equation"][:2] == 'X=':
            data_d["equation"] = data_d["equation"][2:]
        question, equation, answer = data_d["original_text"], data_d["equation"], data_d["ans"]

        if used_equ_norm:
            Transfer = TransferNum(dataset='math23k')
            trans_equ, equation = Transfer.normalized_equation(data_d['segmented_text'], equation, norm_count)
        else:
            trans_equ = None

        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', str(answer))
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 处理百分数
        question = re.sub('([\.\d]+)%', '(\\1/100)', question)  # add by sfeng
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 分数去括号
        # question = re.sub('\((\d+/\d+)\)', '\\1', question)
        question = re.sub('\(([0-9]+\.?[0-9]*)/([0-9]+\.?[0-9]*)\)', '\\1/\\2', question)
        # 冒号转除号、百分号转除号
        equation = equation.replace(':', '/').replace('%', '/100').replace('[', '(').replace(']', ')')
        answer = answer.replace(':', '/').replace('%', '/100')

        if mode == 'train':
            if single_char:
                equation = Prefix.expression2prefix(equation, answer, data_d, single_char=single_char, dataset=dataset)
            else:
                pre_equation = Prefix.expression2prefix(equation, answer, data_d, single_char=single_char, dataset=dataset)
                if pre_equation is not None:
                    if split_number:
                        equation = Prefix.split_decimal(pre_equation)
                    else:
                        equation = pre_equation
                    equation = str(equation)
                else:
                    equation = None
        if equation is not None:
            seg_text_char = self.partition(question)
            data_d["seg_text_char"] = seg_text_char
            data_d["equation"] = equation
            data_d["ans"] = answer
            data.append(data_d)
        return data, trans_equ

    def load_math23k_line_data(self, filename, mode='train', single_char=True, used_equ_norm=False):
        print("Reading lines...")
        f = open(filename, encoding="utf-8")
        js = ""
        data = []
        equ_ids_dict = defaultdict(list)
        norm_count = 0
        trans_equations, after_norm_equs = [], []
        for i, s in enumerate(f):
            js += s
            i += 1
            if i % 7 == 0:  # every 7 line is a json
                data_d = json.loads(js,strict=False)
                data, trans_equ = self.process_input(data_d, data, mode, single_char, norm_count=norm_count, used_equ_norm=used_equ_norm)
                if used_equ_norm:
                    norm_count, trans_equation, after_norm_equ = trans_equ
                    trans_equations.append(trans_equation)
                    after_norm_equs.append(after_norm_equ)
                    equ_ids_dict[after_norm_equ].append(data_d['id'])
                js = ""
        print('norm_count:{}'.format(norm_count))
        print('len of trans_equations:{}, len of equ_had_norms:{}'.format(len(set(trans_equations)), len(set(after_norm_equs))))
        
        return data, equ_ids_dict


class fetch_ape210k(SplitQuestion):

    def load_raw_data(self, filename, mode='train', single_char=True, used_equ_norm=False):
        """读取训练数据，并做一些标准化，保证equation是可以eval的
        参考：https://kexue.fm/archives/7809
        """
        Prefix = PREFIX()
        data = []
        norm_count=0
        trans_equations, after_norm_equs = [], []
        equ_ids_dict = defaultdict(list)
        for l in open(filename, 'r', encoding='utf-8'):
            data_d = json.loads(l)
            question, equation, answer = data_d['original_text'], data_d['equation'], data_d['ans']
            if equation[:2] == 'x=':
                equation = equation[2:]

            if used_equ_norm:
                Transfer = TransferNum(dataset='ape210k')
                trans_equ, equation = Transfer.normalized_equation(data_d['segmented_text'], equation, norm_count)
                norm_count, trans_equation, after_norm_equ = trans_equ
                trans_equations.append(trans_equation)
                after_norm_equs.append(after_norm_equ)
                equ_ids_dict[after_norm_equ].append(data_d['id'])

            # 处理带分数
            question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
            equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
            answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
            equation = re.sub('(\d+)\(', '\\1+(', equation)
            answer = re.sub('(\d+)\(', '\\1+(', answer)
            # 处理百分数
            question = re.sub('([\.\d]+)%', '(\\1/100)', question)  # add by sfeng
            equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
            answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
            # 分数去括号
            question = re.sub('\((\d+/\d+)\)', '\\1', question)
            # 冒号转除号、剩余百分号处理
            pri_equation = equation.replace(':', '/').replace('%', '/100').replace('[', '(').replace(']', ')').replace('**', '^')
            answer = answer.replace(':', '/').replace('%', '/100')

            if mode == 'train':
                if single_char:
                    equation = Prefix.expression2prefix(pri_equation, answer, single_char=True)
                else:
                    pre_equation = Prefix.expression2prefix(pri_equation, answer, single_char=False)
                    if pre_equation is not None:
                        equation = []
                        for eq in pre_equation:
                            if isnumber(eq) and '.' in eq:
                                equation.extend([eq[:eq.index('.')], '.', eq[eq.index('.')+1:]])
                            else:
                                equation.append(eq)
                        equation = str(equation)
                    else:
                        equation = None
                if equation is not None and is_equal(simplify(pri_equation), eval(answer)):
                    data_d["original_text"] = question
                    data_d["seg_text_char"] = self.partition(question)
                    data_d["equation"] = equation
                    data_d["ans"] = answer
                    data.append(data_d)
            else:
                data_d["original_text"] = question
                data_d["seg_text_char"] = self.partition(question)
                data_d["equation"] = pri_equation
                data_d["ans"] = answer
                data.append(data_d)
        print('norm_count:{}'.format(norm_count))
        print('len of trans_equations:{}, len of equ_had_norms:{}'.format(len(set(trans_equations)), len(set(after_norm_equs))))
        return data, equ_ids_dict


class TransferNum:
    def __init__(self, dataset):
        self.pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
        self.ques_number_digit = {}
        self.ques_digit_number = {}
        self.dataset = dataset
        if self.dataset == 'mawps':
            self.digits = ['aa','bb','cc','dd','ee','ff','ii','jj','ll','mm','pp','rr','ss','tt','xx']
        else:
            self.digits = ['temp_'+s for s in 'abcdefghijklmnopqrstuvwxyz!@#$%^&*']

    def trans_number_digit(self, seg):
        """转换question中的数字为字母"""
        input_seq = []
        nums = []
        for s in seg:
            pos = re.search(self.pattern, s)
            if pos and pos.start() == 0:
                number = s[pos.start(): pos.end()]
                if self.dataset == 'mawps':
                    number = float(number)
                if number not in nums:
                    digit = self.digits[len(nums)]
                    nums.append(number)

                    self.ques_number_digit[number] = digit
                    self.ques_digit_number[digit] = number
                    input_seq.append(digit)
                else:
                    input_seq.append(self.ques_number_digit[number])
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        return input_seq

    def split_equ_str2list(self, equations):
        """分割字符串形式的表达式，转换成list，其中数字归并到一起"""
        equations_split = []
        st = ''
        length_equations = len(equations)
        for i, tok in enumerate(equations):
            if tok in ['+', '-', '*', '/', '^', '(', ')']:
                equations_split.append(tok)
            else:
                st += tok
                if i+1 < length_equations and equations[i+1] in ['+', '-', '*', '/', '^', '(', ')']:
                    equations_split.append(st)
                    st = ''
                if i == length_equations - 1:
                    equations_split.append(st)
        return equations_split

    def transfer_num(self, data):  # transfer num into "NUM"        
        equations = data["equation"]
        equations = self.split_equ_str2list(equations)
        new_equations = []
        for equ in equations:
            if equ in self.ques_number_digit:
                new_equations.append(self.ques_number_digit[equ])
            else:
                new_equations.append(str(equ))
        data["trans_equation"] = str(new_equations)
        data['ques_digit_number'] = str(self.ques_digit_number)
        return data

    def preprocess_input(self, segmented_text, primary_equation):
        if self.dataset == 'ape210k':
            segmented_text = re.sub('\( (\d+)\s+(/)\s+([0-9]+\.?[0-9]*) \)', '(\\1/\\3)', segmented_text)  # ( 1 / 5 ) -> (1/5)
            segmented_text = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', segmented_text)
            segmented_text = re.sub('(\d+)\(', '\\1+(', segmented_text)
            segmented_text = segmented_text.replace('**', '^').replace(':', '/')
            primary_equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', primary_equation)
            primary_equation = re.sub('(\d+)\(', '\\1+(', primary_equation)
            primary_equation = primary_equation.replace('**', '^').replace(':', '/')
        seg = segmented_text.strip().split(" ")
        input_seq = self.trans_number_digit(seg)
        primary_equation = primary_equation.replace('[', '(').replace(']', ')')
        equations = self.split_equ_str2list(primary_equation)
        submerge_equations = []
        i = 0
        while i < len(equations):
            if equations[i] == '(' and (i+4) < len(equations) and ''.join(equations[i:i+5]) in seg:
                submerge_equations.append(''.join(equations[i:i+5]))
                i += 5
            else:
                submerge_equations.append(equations[i])
                i += 1
        return submerge_equations, segmented_text, primary_equation

    def normalized_equation(self, segmented_text, primary_equation, norm_count):
        """"表达式归一化，同时处理分式情况"""
        submerge_equations, segmented_text, primary_equation = self.preprocess_input(segmented_text, primary_equation)
        trans_equation = [self.ques_number_digit[equ] if equ in self.ques_number_digit else str(equ) for equ in submerge_equations]
        equ_had_norm = norm_equ(trans_equation)
        equ_filter_repeat = filter_repeat_num(equ_had_norm)
        new_equation = [self.ques_digit_number[equ] if equ in self.ques_digit_number else equ for equ in equ_filter_repeat]
        if new_equation == []:
            new_equation.append('0')
        try:
            assert is_equal(eval(''.join(new_equation).replace('^', '**').replace('%', '/100')), eval(primary_equation.replace('^', '**').replace('%', '/100')))
        except:
            print('...'*30)
            print('primary_equation:{}'.format(primary_equation))
            print('segmented_text:{}'.format(segmented_text))
            print('...'*30)

        if trans_equation != equ_filter_repeat:
            # print('primary_equation:{}, trans_equation:{}, equ_had_norm:{}, equ_filter_repeat:{}'.format(primary_equation, trans_equation, equ_had_norm, equ_filter_repeat))
            norm_count += 1
        return [norm_count, ''.join(trans_equation), ''.join(equ_filter_repeat)], ''.join(new_equation)


    

