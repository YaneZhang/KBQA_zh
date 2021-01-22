from concurrent import futures
import logging

from qa_system import find_answer

import grpc
import time
import qa_pipeline_pb2_grpc
import qa_pipeline_pb2

_ONE_DAY_IN_SECONDS = 60*60*24


class Greeter(qa_pipeline_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        text = request.text
        print(text)
        pred = find_answer(text)
        # pred = text
        return qa_pipeline_pb2.HelloReply(message='%s' % pred)


def server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    qa_pipeline_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
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