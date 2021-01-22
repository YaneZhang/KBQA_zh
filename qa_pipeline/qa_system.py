'''
Author: yanpengzhang
Date: 2021-01-22 11:29:58
FilePath: /KBQA_zh/qa_pipeline/qa_system.py
'''
from ner_module import ner_predict
from qa_module import qa_predict

import pymysql

username = 'username'
password = 'password'
host = 'your_db_host'
db = 'db'


def create_cursor(host, username, password, db):
    conn = pymysql.connect(host, username, password, db)
    return conn.cursor()

cursor = create_cursor(host, username, password, db)

def find_answer(question):

    entities = ner_predict(question)
    if not entities:
        return "sorry, 这个问题我没有理解您想做什么, 正在帮您转接人工...\n"
    entity = entities[0]

    sql = f"select predicate, object from kbqa_triples where subject = '{entity}'"
    cursor.execute(sql)
    pre_obj_pairs = cursor.fetchall()
    if not pre_obj_pairs:
        return "sorry, 我的知识库还在丰富中, 这个问题正在帮您转接人工...\n"

    scores = []
    for i, (predicate, obj) in enumerate(pre_obj_pairs):
        score = qa_predict(question, predicate)
        scores.append([score, predicate, obj])
    
    scores.sort(key=lambda x: x[0], reverse=True)

    if scores[0][0] < 0.5:
        return "这个问题有点难，正在帮您转接人工...\n"
    if len(scores)>=3:
        answer = f'''
                    答案：{scores[0][2]}
                    您可能还会对以下咨询感兴趣：
                    1. {scores[1][1]}
                    2. {scores[2][1]} 
                    '''
    else:
        answer = f'''
                    答案：{scores[0][1]}
                    '''
    return answer

if __name__ == "__main__":
    print("hello, 我是yaner, 如果想要结束对话, 请输入end, 让我们开启一次完美的QA之旅吧, let's go !!!")
    while 1:
        q = input("请提问：\n")
        if q == "end":
            print("byebye~")
            print('''
                   ┌─┐       ┌─┐
                ┌──┘ ┴───────┘ ┴──┐
                │                 │
                │       ───       │
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
        answers = find_answer(q,cursor)
        print(answers)

