import ctypes  # 导入ctypes库，用于与Windows API进行交互
import pyautogui  # 导入pyautogui库，用于模拟鼠标操作
import ultralytics  # 导入ultralytics库，用于实现目标检测功能
import pynput  # 导入pynput库，用于监听鼠标事件

# 定义全局变量
mouse_click_flag = False  # 定义鼠标点击标志，初始值为False
sc_w, sc_h = pyautogui.size()  # 获取屏幕宽度（sc_w）和高度（sc_h）
w, h = 700, 1000  # 定义要截取的屏幕宽度和高度
sc_center_x, sc_center_y = sc_w // 2, sc_h // 2  # 计算屏幕的中心点坐标
x, y = sc_center_x - w // 2, sc_center_y - h // 2  # 计算要截取屏幕的左上角坐标

# 定义左键点击函数
def left_click(x, y):
    x = int(x)  # 将坐标转换为整数
    y = int(y)  # 将坐标转换为整数
    ctypes.windll.user32.SetCursorPos(x, y)  # 设置鼠标位置
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # 鼠标左键按下
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # 鼠标左键释放

# 定义预测初始化函数
def predict_init():
    global model  # 定义全局变量model
    model = ultralytics.YOLO("best.pt", task="detect")  # 加载预训练模型

# 定义预测函数
def predict(img):
    global model  # 定义全局变量model
    res = model(source=img, verbose=False, iou=0.6, conf=0.6)  # 进行目标检测
    return res[0]  # 返回检测结果

# 定义监听鼠标函数
def listen_mouse(x, y, button, pressed):
    global mouse_click_flag  # 定义全局变量mouse_click_flag
    if button == pynput.mouse.Button.right:  # 如果按下的是右键
        if pressed:  # 如果右键被按下
            mouse_click_flag = True  # 设置鼠标点击标志为True
        if not pressed:  # 如果右键被释放
            mouse_click_flag = False

# 如果直接运行此文件，则执行以下代码
if __name__ == "__main__":
    predict_init()  # 调用预测初始化函数

    mouse_listener = pynput.mouse.Listener(on_click=listen_mouse)  # 创建鼠标监听器
    mouse_listener.start()  # 启动鼠标监听器

    while True:
        img = pyautogui.screenshot(region=(x, y, w, h))  # 截取屏幕

        predict_res = predict(img)  # 对截取的屏幕进行目标检测
        boxes = predict_res.boxes  # 获取检测结果中的边界框

        if mouse_click_flag:  # 如果鼠标点击标志为True
            if len(boxes) > 0:  # 如果检测到了目标
                left_click(boxes.xywh[0][0].cpu().numpy() + x,
                                    boxes.xywh[0][1].cpu().numpy() + y)
