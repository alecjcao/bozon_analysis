import socket
import threading
import logging
import struct
import time
import json
from PyQt5.QtCore import QObject, pyqtSignal

import platform
if platform.node() == 'GLaDOS':
    HOST = 'localhost'
elif platform.node() == 'burrito':
    HOST = '192.168.59.41' #local lab intranet
else:
    HOST = 'localhost'
PORT = 12345
CLIENT_NAME = 'analysis'

HEARTBEAT_INTERVAL = 1
HEARTBEAT_TIMEOUT = 5
RECONNECT_INTERVAL = 1

class SocketHandler(QObject):
    """
    Handles communication with bozon_manager server for automatically executing data analysis.
    """
    socket_status = pyqtSignal()

    def __init__(self, data_handler, image_processor, analysis_handler):
        super().__init__()

        self.data_handler = data_handler
        self.image_processor = image_processor
        self.analysis_handler = analysis_handler
        
        self.running = False
        self.connected = False
        self.socket = None
        self.client_thread = threading.Thread(target=self.connect_and_listen, daemon=True)
        self.client_thread.start()
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_msg(self, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        self.socket.sendall(msg)

    def recv_msg(self):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def start(self):
        """Starts the socket connection and listening thread."""
        if not self.running:
            self.running = True

    def stop(self):
        """Stops the socket handler and closes the connection."""
        self.running = False

    def connect_and_listen(self):
        """Handles connection, reconnection, and message listening."""
        while True:
            try:
                if not self.running:
                    raise ConnectionError
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(HEARTBEAT_TIMEOUT)
                self.socket.connect((HOST, PORT))
                self.send_msg(format_message('id'))

                self.connected = True
                self.socket_status.emit()
                logging.info("Connected to server.")
                while self.running:
                    data = self.recv_msg()
                    if not data:
                        raise ConnectionError
                    msg = json.loads(data.decode('utf-8'))
                    if not msg['type'] == 'heartbeat':
                        logging.info(f'Received ' + msg['type'] + ' from server.')
                        threading.Thread(target=self.respond, args = (msg,), daemon=True).start()

            except (ConnectionError, socket.error) as e:
                if self.connected:
                    if self.running:
                        logging.warning("Server disconnected.")
                    else:
                        logging.info("Disconnected from server.")
                    self.connected = False
                    self.socket_status.emit()
                    self.socket.close()
                time.sleep(RECONNECT_INTERVAL)  # Wait before retrying

    def heartbeat(self):
        while True:
            try:
                if self.connected and self.running:
                    self.send_msg(format_message('heartbeat'))
                time.sleep(HEARTBEAT_INTERVAL)
            except (ConnectionError, socket.error):
                pass

    
    def respond(self, msg_in):
        result = {}
        processed_images = True
        if msg_in['type'] == 'analyze data':
            self.data_handler.date = self.data_handler.get_most_recent_date()
            self.data_handler.file = self.data_handler.get_most_recent_file()
            try:
                self.image_processor.process_images()
            except Exception as e:
                logging.error(f"Unexpected error in processing images: {e}")
                processed_images = False
            if processed_images and msg_in['data']:
                try:
                    self.analysis_handler.module_name = msg_in['data']['analysis_script']
                    result = self.analysis_handler.run_analysis_script()
                except KeyError:
                    logging.error("Did not find analysis script in received message.")
                except Exception as e:
                    logging.error(f"Unexpected error in analysis script: {e}")
        if self.connected:
            logging.info(f"Sending result {json.dumps(result)} to server.")
            self.send_msg(format_message('update', result))

@staticmethod
def format_message(type, data = None):
    return json.dumps({'from' : CLIENT_NAME, 'type' : type, 'data' : data}).encode('utf-8')