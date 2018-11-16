import socket


def main():
    host = '127.0.0.1'
    port = 5000

    # Request the connection with the listening server.
    # Bind a socket to current machine
    s = socket.socket()
    s.bind((host, port))

    # Start listening one connection.
    s.listen(1)
    # Accept the connection - store the client, that we accept
    c, addr = s.accept()

    print("Connection from", str(addr))

    while True:
        # Transfer data between client and server.
        data = c.recv(1024).decode('utf-8')
        if not data:
            break
        print("From connected user: ", data)
        data = data.upper()
        print("Sending data:", data)
        c.send(data.encode('utf-8'))
    # Close a socket at the end of program.
    c.close()


if __name__ == '__main__':
    main()
