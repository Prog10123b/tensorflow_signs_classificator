# v1.0.0
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import time
import sys,os

from image_classification import classificator

classific = classificator()

classific.load_entire_model('save\\save_cnn_1')

cap = cv.VideoCapture(1) #поток с камеры

uh = 132 #создаем переменные для фильтра HSV
us = 255
uv = 255
lh = 85
ls = 40
lv = 40
thr_f = 70
thr_l = 70
thr_r = 70
thr_fl = 70
thr_fr = 70

lower_hsv = np.array([lh,ls,lv]) #пакуем эти переменные в массив
upper_hsv = np.array([uh,us,uv])

window_name = "detector_calibration" #название окна
cv.namedWindow(window_name) #создаем окно

window_thresholds = "thresholds" #название окна
cv.namedWindow(window_thresholds) #создаем окно

def nothing(x): #коллбек функция на изменение положения ползунка
    print("Trackbar value: " + str(x))
    pass

# создаем трекбары для Upper HSV
cv.createTrackbar('UpperH',window_name,0,255,nothing)
cv.setTrackbarPos('UpperH',window_name, uh)

cv.createTrackbar('UpperS',window_name,0,255,nothing)
cv.setTrackbarPos('UpperS',window_name, us)

cv.createTrackbar('UpperV',window_name,0,255,nothing)
cv.setTrackbarPos('UpperV',window_name, uv)

# создаем трекбары Lower HSV
cv.createTrackbar('LowerH',window_name,0,255,nothing)
cv.setTrackbarPos('LowerH',window_name, lh)

cv.createTrackbar('LowerS',window_name,0,255,nothing)
cv.setTrackbarPos('LowerS',window_name, ls)

cv.createTrackbar('LowerV',window_name,0,255,nothing)
cv.setTrackbarPos('LowerV',window_name, lv)

cv.createTrackbar('forward',window_thresholds,0,100,nothing)
cv.setTrackbarPos('forward',window_thresholds, thr_f)
cv.createTrackbar('left',window_thresholds,0,100,nothing)
cv.setTrackbarPos('left',window_thresholds, thr_l)
cv.createTrackbar('right',window_thresholds,0,100,nothing)
cv.setTrackbarPos('right',window_thresholds, thr_r)
cv.createTrackbar('forward_left',window_thresholds,0,100,nothing)
cv.setTrackbarPos('forward_left',window_thresholds, thr_fl)
cv.createTrackbar('forward_right',window_thresholds,0,100,nothing)
cv.setTrackbarPos('forward_right',window_thresholds, thr_fr)

# создаем SimpleBlobdetector с параметрами по умолчанию.     
detector = cv.SimpleBlobDetector_create()

count = 0

def sortByLength(inputStr):
        return len(inputStr)


# начинаем
while(True):
    # захватываем изображение (frame) из видеопотока
    ret, frame = cap.read()


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     #преобразуем изображение в ХСВ
    # hsv = cv.blur(hsv,(5,5))

    mask = cv.inRange(hsv, lower_hsv, upper_hsv)   #применяем фильтр ХСВ и записываем результат в изображение (mask)

    #обновляем положение трекбаров и записываем в переменные
    uh = cv.getTrackbarPos('UpperH',window_name)   
    us = cv.getTrackbarPos('UpperS',window_name)
    uv = cv.getTrackbarPos('UpperV',window_name)
    lh = cv.getTrackbarPos('LowerH',window_name)
    ls = cv.getTrackbarPos('LowerS',window_name)
    lv = cv.getTrackbarPos('LowerV',window_name)
    thr_f = cv.getTrackbarPos('forward',window_thresholds)
    thr_l = cv.getTrackbarPos('left',window_thresholds)
    thr_r = cv.getTrackbarPos('right',window_thresholds)
    thr_fl = cv.getTrackbarPos('forward_left',window_thresholds)
    thr_fr = cv.getTrackbarPos('forward_right',window_thresholds)
    upper_hsv = np.array([uh,us,uv])
    lower_hsv = np.array([lh,ls,lv])

    # находим только ВНЕШНИЕ контуры  на изображении (mask), внтуренние контуры отбрасываются
    contours0, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # создаем пустое изображение (читай пустой массив 5х5х3)
    crop_ellipse = np.zeros((5,5,3), np.uint8)
    if contours0 == None:
        continue

    results = [[],[],[],[],[]]
    # print(results)
    contours0 = sorted(contours0,reverse=True,key=sortByLength)
    for cnt in contours0: #проходимся по всем найденным контурам
        if len(cnt)>10:    #если контур содержит больше четырех точек (не мелкий)
            ellipse = cv.fitEllipse(cnt) #то ищем в контуре эллипс
            # print(ellipse)
            if ellipse: # если эллипс найден
                x = int(ellipse[0][0]) #координата центра эллипса по х
                y = int(ellipse[0][1]) #координата центра эллипса по у
                w = int(ellipse[1][0]) #ширина эллипса
                h = int(ellipse[1][1]) #высота эллипса
                #angle = ellipse[2] #угол наклона (в программе не используется)

                #ratio_ellipse = соотношение сторон эллипса, помогает отбросить вытянутые эллипсы, которые явно не являются знаками
                #ellipse[1] это массив из двух значений ширины и высоты, поэтому максимальное значение массива делится на минимальное, что из этого будет шириной или высотой - значения не имеет
                ratio_ellipse = max(ellipse[1])/(min(ellipse[1])+0.1) # + 0.1 чтобы избежать деления на 0

                #тут начинается веселье
                if (w>20 and h>20 and ratio_ellipse<1.3): #если эллипс больше 20х20 пикселей и не очень вытянутый
                    crop_ellipse = frame[int(y-(h/2) - 8):int(y+(h/2) + 8), int(x-(w/2) - 8):int(x+(w/2) + 8)] #создаем обрезанное изображение маски (crop_ellipse), вырезается из чб маски в месте найденного эллипса
                    if np.count_nonzero(crop_ellipse): #если обрезание удалось

                        # we don't need this because cnn works in 3-channels images
                        # crop_ellipse = cv.cvtColor(crop_ellipse, cv.COLOR_BGR2GRAY)

                        img_temp = cv.resize(crop_ellipse, (32, 32))    # updated size of image (28x28 -> 32x32)

                        predictions = classific.model.predict(np.array([img_temp, ]))
                        class_number = int(np.argmax(predictions[0]))
                        cv.imshow('neural', img_temp)

                        # print("find!")
                        r = [predictions[0][class_number], class_number, ellipse]

                        cv.ellipse(frame,ellipse, (0,255,0),2)
                        cv.putText(frame, str(class_number), (int(ellipse[0][0]), int(ellipse[0][1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 255, 0), 1, cv.LINE_AA)

                        # cv.imshow("ellipse",crop_ellipse)
                        # cv.imshow("forward_img",forward_img_temp)
                        # cv.imshow("match",res_forward_img)

                        # print("empty")

                    else:
                        #если обрезание не удалось, то рисуем пустоту
                        crop_ellipse = np.zeros((500,500,3), np.uint8) 
                        continue #и переходим к следующему эллипсу

                else:
                    #если обрезание не удалось, то рисуем пустоту
                    crop_ellipse = np.zeros((500,500,3), np.uint8) 
                    continue #и переходим к следующему эллипсу
            else:
                #если эллипс не найден, то тоже рисуем пустоту
                crop_ellipse = np.zeros((500,500,3), np.uint8) 
                continue #и ищем следующий эллипс
        # else:
            # break
    

    # print(len(results[0]))

    # if results:
    #     for i in results:
    #         # for e in i:
    #         #     cv.ellipse(frame,e[2],(0,255,0),2)
    #         #     cv.putText(frame,str(e[0]),(int(e[2][0][0]),int(e[2][0][1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1, cv.LINE_AA)
    #
    #         results_r = zip(*i[::-1])
    #         try:
    #             # print(results_r)
    #             num_fin = results_r[0].index(max(results_r[0]))
    #             cv.ellipse(frame,results_r[2][num_fin],(0,0,255),2)
    #             cv.putText(frame,str(results_r[1][num_fin]),(int(results_r[2][num_fin][0][0]),int(results_r[2][num_fin][0][1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1, cv.LINE_AA)
    #         except:
    #             pass

    cv.imshow(window_name,mask) #выводим окно с настройками ХСВ и большую маску   
    cv.imshow("original",frame) #рисуем исходное изображение
    
    if cv.waitKey(1) & 0xFF == ord('q'): #если нажата кнопка q на клавиатуре, то завершить цикл
        break

cap.release() #остановка видеопотока
cv.destroyAllWindows() #закрываем все окна
