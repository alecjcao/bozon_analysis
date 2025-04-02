import socket
import threading
import logging
import struct
import time
import json
from PyQt5.QtCore import QObject, pyqtSignal

import platform
if platform.node() == 'GLaDOS':
    HOST = '127.0.0.1'
elif platform.node() == 'burrito':
    HOST = ''
else:
    HOST = '127.0.0.1'
PORT = 12345
CLIENT_NAME = 'analysis'

class SocketHandler(QObject):
    socket_status = pyqtSignal()

    def __init__(self, data_handler, image_processor, analysis_handler):
        super().__init__()

        self.data_handler = data_handler
        self.image_processor = image_processor
        self.analysis_handler = analysis_handler
        
        self.running = False
        self.connected = False
        self.socket = None
        self.thread = None
        self.reconnect_interval = 1

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
            self.socket_status.emit()
            self.thread = threading.Thread(target=self.connect_and_listen, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the socket handler and closes the connection."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.socket:
            self.socket.close()
            self.socket = None
        self.socket_status.emit()

    def connect_and_listen(self):
        """Handles connection, reconnection, and message listening."""
        while self.running:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1)
            try:
                self.socket.connect((HOST, PORT))
                self.send_msg(CLIENT_NAME.encode('utf-8'))

                self.connected = True
                self.socket_status.emit()
                while self.running:
                    try:
                        data = self.recv_msg()
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionResetError("Server closed the connection.")
                    msg_in = data.decode('utf-8')
                    msg_out = self.respond(msg_in)
                    self.send_msg(msg_out.encode('utf-8'))

            except (ConnectionRefusedError, ConnectionResetError, socket.error) as e:
                self.connected = False
                self.socket_status.emit()
                time.sleep(self.reconnect_interval)  # Wait before retrying
        self.running = False
        self.connected = False
        self.socket.close()
        self.socket = None
    
    def respond(self, msg_in):
        result = {}
        self.data_handler.get_most_recent_date()
        self.data_handler.get_most_recent_file()
        try:
            self.image_processor.process_images()
        except Exception as e:
            logging.error(f"Error in processing images: {e}")
            return json.dumps(result)
        try:
            msg_in = json.loads(msg_in)
            self.analysis_handler.module_name = msg_in['analysis_script']
        except ValueError:
            logging.error("Invalid non-JSON message received.")
            return json.dumps(result)
        except KeyError:
            logging.error("Did not find analysis script in received message.")
            return json.dumps(result)
        try:
            result = self.analysis_handler.run_analysis_script()
        except Exception as e:
            logging.error(f"Error in analysis script: {e}")
        return json.dumps(result)

