import socket


def main():
    host = '127.0.0.1'
    port = 5000

    s = socket.socket()
    s.connect((host, port))

    # Get message from user
    message = input('>> ')
    while message != 'q':
        # Send data to server
        s.send(message.encode('utf-8'))

        # Receive data back from the server.
        data = s.recv(1024).decode('utf-8')
        print("Received from server:", data)

        # Get message again.
        message = input('>>')
    # At the end of the program, close client socket.
    s.close()


if __name__ == '__main__':
    main()
