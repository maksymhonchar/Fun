import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def main():
    host = '127.0.0.1'
    port = 5001
    server = ('127.0.0.1', 5000)

    # Bind a UDP socket
    s.bind((host, port))

    msg = input(">> ")
    while msg != 'q':
        s.sendto(msg.encode('utf-8'), server)
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')
        print('Message from server:', data)

        # send a message again
        msg = input(">> ")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        s.close()
