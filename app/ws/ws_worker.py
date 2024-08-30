import json
from typing import Optional
import secrets

from .ws_socket import WSSocket, WSData, WSMessage


class WSWorker:

    async def process(self, ws_server,ws_socket, message: Optional[WSMessage]):

        socket_type = ws_socket.type
        socket_client_id = ws_socket.client_id
        message_key = secrets.token_urlsafe(12)

        app_socket = None
        # Load app-socket
        try:
            app_socket = ws_server.WORKERS[socket_client_id]
            print("device connected found: " + socket_client_id)
        except Exception as ex1:
            print("can not find device " + socket_client_id)


       
        # command processing
        
        # if socket_type == 'device':
        #     socket_command = message.data
        #     shop_id = str(message.shop_id)
            
        #     # send request faceid to booth
        #     if socket_command == 'faceid':
        #         if booth_socket is None:
        #             dt = {}
        #             dt['result'] = 'booth-not-ready'
        #             dt['data'] = 'error'
        #             await app_socket.send(dict(type='event', event='faceid_rs', message_key=message_key,data=dt))
        #         else:
        #             await booth_socket.send(dict(type='event', event='start_camera', message_key=message_key,data='start_camera',shop_id=shop_id))


        # elif socket_type == 'booth':
        #     socket_command = message.data["command"]
            
        #     # send faceid_result to device
        #     if socket_command == "faceid_rs":
        #         await app_socket.send(dict(type='event', event='faceid_rs', message_key=message_key, data=message.data))

        print(socket_type + " " + socket_client_id + " " + socket_command)           