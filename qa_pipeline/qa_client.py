'''
Author: yanpengzhang
Date: 2021-01-22 14:50:56
FilePath: /KBQA_zh/qa_pipeline/qa_client.py
'''
from __future__ import print_function
import logging
import sys

import grpc

import qa_pipeline_pb2_grpc
import qa_pipeline_pb2

def run(text):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = qa_pipeline_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(qa_pipeline_pb2.HelloRequest(text=text))
    print(response.message)


if __name__ == '__main__':
    logging.basicConfig()
    print("hello, 我是yaner, 如果想要结束对话, 请输入end, 让我们开启一次完美的QA之旅吧, let's go !!!")
    while 1:
        question = input("请提问：\n")
        if question == "end":
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
        run(question)