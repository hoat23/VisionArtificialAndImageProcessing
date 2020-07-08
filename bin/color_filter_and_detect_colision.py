import cv2
import numpy as np
import random


#Aqui dibujamos el cuadro con un punto inicial random y un punto final random + 50 puntos
def dibujarRec(frame, numero):
    cv2.rectangle(frame, ( numero, numero), (numero + 50, numero + 50), (0, 255, 0), 3)
    return [[numero,numero],[numero+50,numero+50]]

def detectar_color(frame, range_color):
    xr, yr, wr, hr = 0, 0, 0, 0
    kernel_erode = np.ones((4,4),np.uint8)
    kernel_close = np.ones((15,15),np.uint8)
    #converting to hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    upper = np.array(range_color[1])
    lower = np.array(range_color[0])
    mask1 = cv2.inRange(hsv, lower, upper)
    mask1 = cv2.erode(mask1, kernel_erode, iterations=1)
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel_close)
    #encontrando contornos
    _, contour, _ = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #dibujando un rectangulo en el contorno
    try:
        for i in range (0,10):
            xr, yr, wr, hr = cv2.boundingRect(contour[i])
            if (wr*hr)>2000:
                break
    except:
        pass
    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)
    #Calculando el centro
    point_x = int(xr+(wr/2))
    point_y = int(yr+(hr/2))
    #Dibujando el centro
    cv2.circle(frame, (point_x, point_y), 5 , (10, 200, 150), -1)
    #Devuelve los dos puntos superior izquierdo e inferior derecho 
    return [[xr,yr], [xr+wr, yr+hr]]

def punto_dentro_rectagulo(punto, rectangulo):
    [[Ax1,Ay1], [Ax2,Ay2]] = rectangulo
    [Px1,Py1] = punto
    if (Ax1 <= Px1 and Px1<= Ax2) and (Ay1<=Py1 and Py1<=Ay2):
        return True
    return False

def detectar_colision(rect_A, rect_B):
    """
    rect_A = [ [Ax1,Ay1] , [Ax2,Ay2] ] #Puntos extremos
    rect_B = [ [Bx1,By1] , [Bx2,By2] ] #Puntos extremos
    """
    [Bx1,By1] = rect_B[0]
    [Bx2,By2] = rect_B[1]
    puntos_rectangulo_B = [ [Bx1, By1], [Bx2,By1], [Bx2,By2], [Bx1,By2] ]
    for punto in puntos_rectangulo_B:
        if punto_dentro_rectagulo(punto, rect_A):
            return True
    return False

if __name__ == "__main__":    
    video = cv2.VideoCapture(0)
    kernel = np.ones((5,5), np.uint8)
    #Inicializando el tipo de letra
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    #Aqui declaramos la variable "numero" con un rango de 50 a 428 que es el area delimitada de trabjo.
    numero = random.randint (50, 428)

    while (True):
        _,frame = video.read()
        #fliping the frame horizontally.
        frame = cv2.flip(frame,1)
        rect_random = dibujarRec(frame,numero)

        #Aqui estamos detectando el color amarillo, dibujando su contorno y su centro
        rojo_range = [[136, 87, 111],[179, 255, 255]] #amarillo_range = [[20, 80, 80],[40, 255, 255]]#da problemas de deteccion
        rect_objdetect = detectar_color(frame,rojo_range)

        #Aqui nombramos la ventana de visualizacion (vide0)
        cv2.imshow('VIDEO', frame)

        if detectar_colision(rect_random, rect_objdetect):
            video.release()
            cv2.destroyAllWindows()
            cv2.putText(frame, str("Colision"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, str("Press any key to Exit."), (180, 260), font, 1, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame",frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        #Aqui paramos el bucle while presionando la tecla "c"
        if cv2.waitKey(1) & 0xFF == ord ('c'):
            video.release()
            cv2.destroyAllWindows()
            break
        
    pass

