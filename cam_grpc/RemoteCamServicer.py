from concurrent import futures
import logging
import math
import time
import grpc
import RemoteCam_pb2
import RemoteCam_pb2_grpc
from threading import Thread
import cv2
import numpy as np

PORT_CAM = 10509

class RemoteCamServicer(RemoteCam_pb2_grpc.RemoteCamProtoServicer):
    def __init__(self):
        pass # init cam here

    def GetImage(self, request, context):
        request_id = request.request_id
        color = cv2.imread("test-container.png")         # sample image
        depth = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)  # fake depth map with gray image
        return RemoteCam_pb2.GetImageResponse(response_id=request_id,
                                              width=color.shape[1],height=color.shape[0],
                                              color=color.flatten(),
                                              depth=depth.flatten())
        
        
def serve(servicer, host='[::]'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    RemoteCam_pb2_grpc.add_RemoteCamProtoServicer_to_server(
        servicer, server)
    server.add_insecure_port('{}:{}'.format(host, PORT_CAM))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    servicer = RemoteCamServicer()
    serve(servicer)