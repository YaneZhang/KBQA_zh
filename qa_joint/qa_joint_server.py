from concurrent import futures
import logging
import pymysql

from joint_module import joint_predict

import grpc
import time
import qa_joint_pd2_grpc
import qa_joint_pd2

_ONE_DAY_IN_SECONDS = 60*60*24

username = 'username'
password = 'password'
host = 'your_db_host'
db = 'db'

def create_cursor(host, username, password, db):
    conn = pymysql.connect(host, username, password, db)
    return conn.cursor()

cursor = create_cursor(host, username, password, db)

def find_answer(sentence):
    intent, entities = joint_predict(sentence)
    if not intent or not entities:
        return "sorry, 这个问题我没有理解您想做什么, 正在帮您转接人工...\n"

    sql = f"select object from kbqa_triples where subject = '{entities[0]}' and predicate = '{intent}'"
    cursor.execute(sql)
    obj = cursor.fetchall()

    if not obj:
        return "sorry, 我的知识库还在丰富中, 这个问题正在帮您转接人工...\n"

    ans = f"答案：{obj[0]}"
    return ans
        

class Greeter(qa_joint_pd2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        text = request.text
        print(text)
        pred = find_answer(text)
        return qa_joint_pd2.HelloReply(message=pred)


def server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    qa_joint_pd2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    # server.wait_for_termination()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    logging.basicConfig()
    server()