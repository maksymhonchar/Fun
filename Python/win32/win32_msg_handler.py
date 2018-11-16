import win32con
import win32api
import win32gui
import time


mainloop_delay = 0.1  # [s]

className = "CLICK PLC manager WM handler className"
windowName = "CLICK PLC manager WM handler"

WM_START_ACCUMUL_DATA = win32con.WM_USER + 0
WM_STOP_ACCUMUL_DATA  = win32con.WM_USER + 1


class WMHandler(object):

    def __init__(self, gui_instance):
        self.gui_instance = gui_instance
        self._on_destroy_called = False
    
    def mainloop(self):
        self._build_window()
        while True and not self._on_destroy_called:
            _, msg = win32gui.PeekMessage(None, 0, 0, win32con.PM_REMOVE)
            win32gui.TranslateMessage(msg)
            win32gui.DispatchMessage(msg)
            time.sleep(mainloop_delay)
        win32gui.PostQuitMessage(0)
    
    def _build_window(self):
        msgMap = {
            win32con.WM_DESTROY:    self.on_destroy,
            WM_START_ACCUMUL_DATA:  self.on_start_accumul_data,
            WM_STOP_ACCUMUL_DATA:   self.on_stop_accumul_data
        }        
        hinst = win32api.GetModuleHandle(None)
        wndClass = win32gui.WNDCLASS()
        wndClass.hInstance = hinst
        wndClass.lpfnWndProc = msgMap
        wndClass.lpszClassName = className
        registeredWndClass = win32gui.RegisterClass(wndClass)
        hwnd = win32gui.CreateWindowEx(
            0,  # Extended window style.
            registeredWndClass,  # Class atom created by RegisterClass.
            windowName,  # The window name.
            0,  # Style of the window being created.
            0,  # Initial horizontal position.
            0,  # Initial vertical position.
            0,  # Width of the window.
            0,  # Height of the window.
            win32con.HWND_MESSAGE,  # Message-only window.
            0,  # A handle to a menu.
            hinst,  # A handle to the instance of the main module.
            None  # A pointer to a value for CREATESTRUCT structure.
        )

    def on_destroy(self, hwnd, message, wparam, lparam):
        self._on_destroy_called = True

    def on_start_accumul_data(self, hwnd, message, wparam, lparam):
        self.gui_instance._start_pause_event()

    def on_stop_accumul_data(self, hwnd, message, wparam, lparam):
        self.gui_instance._end_pause_event()
