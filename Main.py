import ctypes
import pyautogui
import ultralytics
import pynput

mouse_click_flag = False
sc_w, sc_h = pyautogui.size()
w, h = 700, 1000
sc_center_x, sc_center_y = sc_w // 2, sc_h // 2
x, y = sc_center_x - w // 2, sc_center_y - h // 2

def left_click(x, y):
    x = int(x)
    y = int(y)
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # 鼠标左键按下
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # 鼠标左键释放
def predict_init():
    global model
    model = ultralytics.YOLO("best.pt", task="detect")

def predict(img):
    global model
    res = model(source=img, verbose=False, iou=0.6, conf=0.6)
    return res[0]

def listen_mouse(x, y, button, pressed):
    global mouse_click_flag
    if button == pynput.mouse.Button.right:
        if pressed:
            mouse_click_flag = True
        if not pressed:
            mouse_click_flag = False

if __name__ == "__main__":
    predict_init()
    
    mouse_listener = pynput.mouse.Listener(on_click=listen_mouse)
    mouse_listener.start()
    
    while True:
        img = pyautogui.screenshot(region=(x, y, w, h))
        
        predict_res = predict(img)
        boxes = predict_res.boxes
        
        if mouse_click_flag:
            if len(boxes) > 0:
                left_click(boxes.xywh[0][0].cpu().numpy() + x, 
                                    boxes.xywh[0][1].cpu().numpy() + y)