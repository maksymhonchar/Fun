import win32gui, win32con, win32api
import array, struct

ext_wm_start = win32con.WM_USER + 0
ext_wm_end   = win32con.WM_USER + 1

hwndMain = win32gui.FindWindow(None, 'CLICK_WMH_WindowName')

def send_message_requests():
    print(hwndMain)
    win32gui.PostMessage(hwndMain, ext_wm_start)
    print('sent ext_wm_start')
    # win32gui.PostMessage(hwndMain, ext_wm_end)
    # print('sent ext_wm_end')

def send_implementation_with_data2(window, message):
    CopyDataStruct = 'IIP'
    dwData = 0x00400001
    buffer = array.array('u', message)
    cds = struct.pack(
        CopyDataStruct, 
        dwData, 
        buffer.buffer_info()[1] * 2 + 1, 
        buffer.buffer_info()[0]
    )
    win32api.SendMesage(window, win32con.WM_USER + 0, 0, cds)

def send_quit_message():
    print('Sending quit message')
    win32gui.PostMessage(hwndMain, win32con.WM_QUIT)
    print('Sending quit destroy')
    win32gui.PostMessage(hwndMain, win32con.WM_DESTROY)

send_message_requests()
# send_quit_message()

print('End of the program.')
