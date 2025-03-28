import socket
import threading
import logging
import struct
import time
import json

import platform
if platform.node() == 'GLaDOS':
    HOST = '127.0.0.1'
elif platform.node() == 'burrito':
    HOST = ''
else:
    HOST = '127.0.0.1'
PORT = 12345

class SocketHandler():
    def __init__(self, image_processor, analysis_handler):
        self.image_processor = image_processor
        self.analysis_handler = analysis_handler
        
        self.running = False
        self.socket = None
        self.thread = None
        self.reconnect_interval = 5

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
            self.thread = threading.Thread(target=self.connect_and_listen, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the socket handler and closes the connection."""
        self.running = False
        if self.socket:
            self.socket.shutdown(socket.SHUT_RDWR)  # Close socket properly
            self.socket.close()
            self.socket = None

        if self.thread and self.thread.is_alive():
            self.thread.join()  # Wait for thread to exit

    def connect_and_listen(self):
        """Handles connection, reconnection, and message listening."""
        while self.running:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((HOST, PORT))

                while self.running:
                    msg_in = self.recv_msg().decode('utf-8')
                    if not msg_in:
                        raise ConnectionResetError("Server closed the connection.")
                    
                    msg_out = self.respond(msg_in)
                    self.send_msg(msg_out.encode('utf-8'))

            except (ConnectionRefusedError, ConnectionResetError, socket.error) as e:
                time.sleep(self.reconnect_interval)  # Wait before retrying
            finally:
                if self.socket:
                    self.socket.shutdown(socket.SHUT_RDWR)  # Close socket properly
                    self.socket.close()
                    self.socket = None
    
    def respond(self, msg_in):
        self.image_processor.process_images()
        try:
            msg_in = json.loads(msg_in)
        except ValueError:
            logging.error("Invalid message received.")
            return json.dumps({})
        self.analysis_handler.module_name = msg_in['analysis_script']
        result = self.analysis_handler.run_analysis_script()
        return json.dumps(result)

