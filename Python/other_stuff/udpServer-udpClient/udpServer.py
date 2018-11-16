import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    host = '127.0.0.1'
    port = 5000

    # Bind a UDP socket
    s.bind((host, port))

    print('Server started')
    while True:
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        print('Message from the', str(addr))
        print('From connected user:', data)

        # Send msg to client
        data = data.upper()
        print('Sending to client:', data)
        s.sendto(data.encode('utf-8'), addr)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        s.close()
