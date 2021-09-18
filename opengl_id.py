from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import pandas as pd

from pyproj import Transformer
from pyproj import CRS

class Stack(object):

    def __init__(self):
        self.stack = []

    def push(self, data):
        """
        进栈函数
        """
        self.stack.append(data)

    def pop(self):
        """
        出栈函数，
        """
        return self.stack.pop()

    def gettop(self):
        """
        取栈顶
        """
        return self.stack[-1]


crs = CRS.from_epsg(4326)
crs_cs = CRS.from_epsg(23870)

transformer = Transformer.from_crs(crs, crs_cs)

IS_PERSPECTIVE = False                              # 透视投影
VIEW = np.array([-50.0, 50.0, -50.0, 50.0, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])                 # 模型缩放比例
EYE = np.array([0.0, 0.0, -1.0])                     # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 2.0])                 # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 640, 480                             # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False                              # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0                             # 考察鼠标位移量时保存的起始位置
frame = 0 
df = None

mainWindow = None
subw1 = None

stack_x = [Stack() for i in range(1000)]
stack_y = [Stack() for i in range(1000)]

record_index_position = 0
#保存车道线的位置
save_position_line_x = [[] for i in range(1000)]
save_position_line_y = [[] for i in range(1000)]

#保存gps的位置
save_gps_x = []
save_gps_y = []

#保存车道线的id
save_lane_id = []


def getposture():
    global EYE, LOOK_AT
    
    dist = np.sqrt(np.power((EYE-LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1]-LOOK_AT[1])/dist)
        theta = np.arcsin((EYE[0]-LOOK_AT[0])/(dist*np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0
        
    return dist, phi, theta
    
DIST, PHI, THETA = getposture()                     # 眼睛与观察目标之间的距离、仰角、方位角

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0) # 设置画布背景色。注意：这里必须是4个参数
    # glEnable(GL_DEPTH_TEST)          # 开启深度测试，实现遮挡关系
    # glDepthFunc(GL_LEQUAL)           # 设置深度测试函数（GL_LEQUAL只是选项之一）

    read_data()

def read_data():
    global df
    df = pd.read_csv("/home/calmcar/zjp/slam_map/data/lane_info.csv")

def read_data_frame(frame):
    global df
    #读取数据列表
    # index = df['index']
    # index = index.values.astype('int')
    line_id = df['line_id']
    line_id = line_id.values.astype('int')

    start_y_list = df['start_y']
    # end_y_list = df['end_y']
    c0_list = df['C0']
    c1_list = df['C1']
    c2_list = df['C2']
    c3_list = df['C3']
    latitude_list = df['latitude']
    longitude_list = df['longitude']
    yaw_list = df['yaw']

    gps_x = []
    gps_y = []

    #收集4条车道线偏移的信息
    lines_x = [[] for i in range(1000)]
    lines_y = [[] for i in range(1000)]

    index_line_id = []
    for n, el in enumerate(line_id[frame:frame+ 30 * 4]):
      index_line_id.append(el)
      if el != 0:
        c0 = c0_list[frame + n]
        c1 = c1_list[frame + n]
        c2 = c2_list[frame + n]
        c3 = c3_list[frame + n]

        latitude = latitude_list[frame + n]
        longitude = longitude_list[frame + n]
        world_x, world_y = transformer.transform(latitude, longitude)
        #world_y += 3.83

        gps_x.append(world_x)
        gps_y.append(world_y)

        #270是否加
        #yaw = yaw_list[frame + n] - 270
        yaw = yaw_list[frame + n] 
        yaw = yaw / 180.0 * np.pi 

        rotate_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                  [np.sin(yaw), np.cos(yaw)]])
        rotate_matrix = np.array([[-1, 0], [0, 1]]).dot(rotate_matrix)
        rotate_matrix_inverse = np.linalg.inv(rotate_matrix)
        
        start_y = start_y_list[frame + n]
        
        x = c0 + c1 * start_y + c2 * start_y * start_y + c3 * start_y * start_y * start_y
        y = start_y

        world_drive_x, world_drive_y = rotate_matrix_inverse.dot(np.array([x, y]))
        
        world_x_copy = world_x + world_drive_x
        world_y_copy = world_y + world_drive_y
        lines_x[el].append(world_x_copy)
        lines_y[el].append(world_y_copy)
      else:
        latitude = latitude_list[frame + n]
        longitude = longitude_list[frame + n]
        
        world_x, world_y = transformer.transform(latitude, longitude)
        
        gps_x.append(world_x)
        gps_y.append(world_y)

        continue

    return lines_x, lines_y, gps_x, gps_y, set(index_line_id)

def draw():
    glutSetWindow(mainWindow)

    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H
    global frame  
    global save_position_line_x 
    global save_position_line_y
    global record_index_position
    global save_gps_x
    global save_gps_y

    # 清除屏幕及深度缓存
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        
    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
        
    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])
    
    position_x,position_y, gps_x, gps_y, index_line_id = read_data_frame(frame)
    
    frame += 30 * 4
    print('--------------', frame)

    EYE[0] = gps_x[0]
    EYE[1] = gps_y[0]                  
    LOOK_AT[0] = gps_x[0]
    LOOK_AT[1] = gps_y[0]

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2], 
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )
    
    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)
    
    #设置颜色
    glColor4f(1.0, 1.0, 1.0, 1.0) 

    glBegin(GL_POINTS)
    for x, y in zip(gps_x, gps_y):
        save_gps_x.append(x)
        save_gps_y.append(y)
        glVertex3f(x,y,0.0)
    glEnd() 
   
    # #设置颜色
    # glColor4f(0.0, 1.0, 0.0, 1.0)
    
    # for i in range(4):
    #   glBegin(GL_LINE_STRIP)
    #   for x, y in zip(fit_x_line[i], fit_y_line[i]):
    #     glVertex3f(x,y,0.0)            
    #   glEnd()

    glColor4f(0.0, 1.0, 1.0, 1.0) 
    
    if save_lane_id:
        fit_x_line = [[] for i in range(1000)]
        fit_y_line = [[] for i in range(1000)]
        for i in save_lane_id[record_index_position:]:
            for x, y in zip(save_position_line_x[i], save_position_line_y[i]):
                fit_x_line[i].append(x)
                fit_y_line[i].append(y) 
    
        for i in save_lane_id[record_index_position:]:
            if not i: continue
            f = np.polyfit(fit_x_line[i], fit_y_line[i], 3)
            p1 = np.poly1d(f)

            max_x = max(fit_x_line[i])
            min_x = min(fit_x_line[i])

            fit_x_line[i] = np.linspace(min_x, max_x, num=10)
            fit_y_line[i] = p1(fit_x_line[i])
            
            stack_x[i].push(fit_x_line[i])
            stack_y[i].push(fit_y_line[i])

        record_index_position += len(save_lane_id) - record_index_position

        for i in save_lane_id:
            glBegin(GL_LINE_STRIP)
            for x, y in zip(fit_x_line[i], fit_y_line[i]):
                glVertex3f(x,y,0.0)
            glEnd()
    
    for i in index_line_id:
      glBegin(GL_LINE_STRIP)
      save_lane_id.append(i)
      for x, y in zip(position_x[i], position_y[i]):
        
        save_position_line_y[i].append(y)
        save_position_line_x[i].append(x)
        glVertex3f(x,y,0.0)
      glEnd()

    glutSwapBuffers()                    # 切换缓冲区，以显示绘制内容
    
def reshape(width, height):
    global WIN_W, WIN_H
    
    WIN_W, WIN_H = width, height
    glutPostRedisplay()
    
def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y
    
    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state==GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()
    
def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H
    
    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y
        
        PHI += 2*np.pi*dy/WIN_H
        PHI %= 2*np.pi
        THETA += 2*np.pi*dx/WIN_W
        THETA %= 2*np.pi
        r = DIST*np.cos(PHI)
        
        EYE[1] = DIST*np.sin(PHI)
        EYE[0] = r*np.sin(THETA)
        EYE[2] = r*np.cos(THETA)
            
        if 0.5*np.pi < PHI < 1.5*np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0
        
        glutPostRedisplay()

def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW
    
    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
        if key == b'x': # 瞄准参考点 x 减小
            LOOK_AT[0] -= 0.01
        elif key == b'X': # 瞄准参考 x 增大
            LOOK_AT[0] += 0.01
        elif key == b'y': # 瞄准参考点 y 减小
            LOOK_AT[1] -= 0.01
        elif key == b'Y': # 瞄准参考点 y 增大
            LOOK_AT[1] += 0.01
        elif key == b'z': # 瞄准参考点 z 减小
            LOOK_AT[2] -= 0.01
        elif key == b'Z': # 瞄准参考点 z 增大
            LOOK_AT[2] += 0.01
        
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\r': # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\x08': # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b' ': # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE 
        glutPostRedisplay()

def changeframe(value):
    #  print(value+1)
    #  draw()
    #  draw1()
    glutPostRedisplay()

    glutTimerFunc(1000, changeframe, value+1)

def draw1():
    glutSetWindow(subw1)
    global save_gps_x
    global save_gps_y

    VIEW = np.array([-50.0, 1000.0, -50.0, 1000.0, 1.0, 20.0])

    # 清除屏幕及深度缓存
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0]*WIN_W/WIN_H, VIEW[1]*WIN_W/WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2]*WIN_H/WIN_W, VIEW[3]*WIN_H/WIN_W, VIEW[4], VIEW[5])
        
    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
        
    # 几何变换
    glScale(1.0, 1.0, 1.0)

    EYE[0] = save_gps_x[0]
    EYE[1] = save_gps_y[0]                  
    LOOK_AT[0] = save_gps_x[0]
    LOOK_AT[1] = save_gps_y[0]

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2], 
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )
    # glViewport(0, 0, WIN_W//2, WIN_H//2)

     #设置颜色
    glColor4f(1.0, 1.0, 1.0, 1.0) 
    glBegin(GL_POINTS)
    for x, y in zip(save_gps_x, save_gps_y):
        # print(x, y)
        glVertex3f(x, y, 0.0)
    glEnd()

    glutSwapBuffers()
    

if __name__ == "__main__":
    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    mainWindow=glutCreateWindow('Quidam Of OpenGL')
    glutTimerFunc(1000, changeframe, 1)
    
    glutSetWindow(mainWindow)
    init()                              # 初始化画布
    glutDisplayFunc(draw)               # 注册回调函数draw()
    glutReshapeFunc(reshape)            # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)           # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)         # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)           # 注册键盘输入的函数keydown()

    # subw1 = glutCreateSubWindow(mainWindow,0,0,320,240)
    # glutSetWindow(subw1)
    # # init()
    # glutReshapeFunc(reshape)            # 注册响应窗口改变的函数reshape()
    # glutMouseFunc(mouseclick)           # 注册响应鼠标点击的函数mouseclick()
    # glutMotionFunc(mousemotion)         # 注册响应鼠标拖拽的函数mousemotion()
    # glutKeyboardFunc(keydown)           # 注册键盘输入的函数keydown()
    # glutDisplayFunc(draw1)

    glutMainLoop()                      # 进入glut主循环