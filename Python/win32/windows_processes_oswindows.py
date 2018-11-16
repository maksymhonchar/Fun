import ctypes
import win32com.client  # pywin32 required.


def find_process(name):
    # More info on theme 'win processes and python': https://mail.python.org/pipermail/python-win32/2003-December/001482.html
    objWMIService = win32com.client.Dispatch('WbemScripting.SWbemLocator')
    objSWbemServices = objWMIService.ConnectServer('.', 'root\\cimv2')
    colItems = objSWbemServices.ExecQuery(
        "Select * from Win32_Process where Caption = '{0}'".format(name)
        # "Select * from Win32_Process"
    )
    return len(colItems)  # colItems

def find_opened_windows():
    # Explanation: https://sjohannes.wordpress.com/2012/03/23/win32-python-getting-all-window-titles/
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    titles = []
    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            titles.append(buff.value)
        return True
    EnumWindows(EnumWindowsProc(foreach_window), 0)
    return titles

# print(find_process('winlogon.exe'))
print(find_opened_windows())
