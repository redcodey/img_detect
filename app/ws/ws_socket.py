import json
from typing import Any, Optional

import websockets
from pydantic.main import BaseModel


class WSMessage(BaseModel):
    type: str
    data: Any = None


class WSData(BaseModel):
    type: str = None
    client_id: Any = None
    socket_id: str
    message: Optional[WSMessage] = None


class WSSocket:

    def __init__(self, type: str, websocket, socket_id: str):
        self.__type = type
        self.__websocket = websocket
        self.__socket_id = socket_id

    async def send(self, msg):
        #print('ready send back')
        if not self.__websocket:
            #print('wse')
            raise websockets.ConnectionClosedError
        await self.__websocket.send(
                WSData(socket_id=self.socket_id, message=msg).json())
        #print(f'sent to client {self.__type} {self.__mac_ip_pub}')

    async def recv(self) -> WSData:
        message = await self.__websocket.recv() 
        if message:
            return WSData.parse_raw(message)
        return None

    @property
    async def messages(self):
        async for message in self.__websocket:
            yield WSData.parse_raw(message)

    @property
    def websocket(self):
        return self.__websocket



    @property
    def open(self):
        if self.__websocket:
            return self.__websocket.open
        return False
