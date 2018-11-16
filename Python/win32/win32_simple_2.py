import win32con, win32api, win32gui
import sys, time


def wndproc(hwnd, msg, wparam, lparam):
    print('wndproc:', msg)


if __name__ == '__main__':
    hinst = win32api.GetModuleHandle(None)
    # Create WNDClASS.
    wndclass = win32gui.WNDCLASS()
    wndclass.hInstance = hinst
    wndclass.lpszClassName = "testWindowClass"
    messageMap = { 
        win32con.WM_QUERYENDSESSION: wndproc,
        win32con.WM_ENDSESSION:      wndproc,
        win32con.WM_QUIT:            wndproc,
        win32con.WM_DESTROY:         wndproc,
        win32con.WM_CLOSE:           wndproc 
    }
    wndclass.lpfnWndProc = messageMap
    # Register WNDCLASS.
    myWindowClass = win32gui.RegisterClass(wndclass)
    hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_LEFT,  # Left-aligned text.
        myWindowClass, 
        "testMsgWindow", 
        0, 
        0, 
        0, 
        win32con.CW_USEDEFAULT, 
        win32con.CW_USEDEFAULT, 
        win32con.HWND_MESSAGE,  # Only as a message handler.
        0, 
        hinst, 
        None
    )
    print('DEBUG: hwnd is', hwnd)

    while True:
        win32gui.PumpWaitingMessages()
        time.sleep(1)
