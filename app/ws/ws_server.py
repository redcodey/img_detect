import json
from typing import Dict

import websockets
import asyncio

from .ws_socket import WSData, WSSocket, WSMessage
from .ws_worker import WSWorker


class WSServer:

    def __init__(self):
        self.WORKERS: Dict[str, WSWorker] = dict()

    def add_worker(self, messge_type: str, worker: WSWorker):
        key = f'{messge_type}'
        #print('addding worker... ', key)
        if key in list(self.WORKERS.keys()):
            raise Exception(f'double worker {key}')
        self.WORKERS[key] = worker

    def remove_worker(self, messge_type: str):
        key = f'{messge_type}'
        if key in list(self.WORKERS.keys()):
            del self.WORKERS[key]

    def worker(self, messge_type: str) -> WSWorker:
        key = f'{messge_type}'
        if key in list(self.WORKERS.keys()):
            return self.WORKERS[key]
        return None

    def broadcast(self, msg):
        sw = set(item.websocket for item in self.WORKERS.values() if item.open)
        websockets.broadcast(sw,
                             WSData(client_id="server", type="server", socket_id="server",
                                    message=msg).json())

    async def handler(self, websocket):
        #print('test handler --> Connected ', websocket)
        message = await websocket.recv()
        #print("msm ", message)
        msg = WSData(**json.loads(message))
        #print('=================================')
        #print('msss', msg)
        ws_socket = WSSocket(msg.type, websocket, msg.socket_id )

        if msg.type == 'booth':
            self.add_boost(ws_socket)
        elif msg.type == 'device':
            #ws_socket.message.__shop_id = msg.shop_id
            self.add_device(ws_socket)
        
        try:
            #print('ready try socket... ', ws_socket.websocket)
            dt = {}
            dt['result'] = "ws-connected"
            dt['data'] = 'success'
            dt['type'] = 'websocket'
            await ws_socket.send(dict(type='event', event='connected', message_key="message_key",data=dt))
            
            async for message in ws_socket.websocket:
                #print('what 2??? ')
                msg = WSData(**json.loads(message))
                #print('fff')
                #print('msg .... ', msg)
                try:
                    #print('ready try socket 2... ', ws_socket.websocket)
                    __worker = self.worker(msg.type)
                    #print("worker ", __worker)
                    if not __worker:
                        __worker = WSWorker()
                    await __worker.process(self, ws_socket, msg.message)
                    #print('seeing process')
                except RuntimeError as exc:
                    await self.error(ws_socket.websocket, str(exc))
                    continue
        except websockets.ConnectionClosedError as cceError:
            #print('what? ', cceError )
            pass
        finally:
            self.remove_boost(ws_socket)
            print('remove socket -- ', ws_socket.type)

    async def error(self, websocket, message):
        event = {
            "type": "error",
            "message": message,
        }
        await websocket.send(json.dumps(event))

    async def run(self, port=80):
        print("ws server start...")
        # async with websockets.serve(self.__handler, "", port):
        #     await asyncio.Future()  # run forever
        loop = asyncio.get_running_loop()
        stop = loop.create_future()
        #loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)
        #port = int(os.environ.get("PORT", "8573"))
        async with websockets.serve(self.handler, "", port):
            await stop

