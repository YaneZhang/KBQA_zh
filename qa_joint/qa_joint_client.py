'''
Author: yanpengzhang
Date: 2021-01-22 14:52:11
FilePath: /KBQA_zh/qa_joint/qa_joint_client.py
'''
from __future__ import print_function
import logging
import sys

import grpc

import qa_joint_pd2_grpc
import qa_joint_pd2

def run(text):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = qa_joint_pd2_grpc.GreeterStub(channel)
        response = stub.SayHello(qa_joint_pd2.HelloRequest(text=text))
    print(response.message)


if __name__ == '__main__':
    logging.basicConfig()
    print("hello, 我是yaner, 如果想要结束对话, 请输入end, 让我们开启一次完美的QA之旅吧, let's go !!!")
    while 1:
        q = input("请提问：\n")
        if q == "end":
            print("byebye~")
            print('''
                   ┌─┐       ┌─┐
                ┌──┘ ┴───────┘ ┴──┐
                │                 │
                │                 │
                │  ─┬┘       └┬─  │
                │                 │
                │       ─┴─       │
                │                 │
                └───┐         ┌───┘
                    │         │
                    │         │
                    │         │
                    │         └──────────────┐
                    │                        │
                    │                        ├─┐
                    │                        ┌─┘
                    │                        │
                    └─┐  ┐  ┌───────┬──┐  ┌──┘
                      │ ─┤ ─┤       │ ─┤ ─┤
                      └──┴──┘       └──┴──┘
                    ''')
            break
        run(q)