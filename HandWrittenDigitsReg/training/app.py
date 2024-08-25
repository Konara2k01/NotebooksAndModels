import pygame,sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

pygame.init()
window_size = (640, 480)

FONT = pygame.font.Font("freesansbold.ttf", 18)

window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Digit board")

# Set up colors
white = (255, 255, 255)
black = (0, 0, 0)
red=(255,0,0)

IMAGESAVE=False
MODEL=load_model("HandWrittenDigitsReg\handWrittenDigitReg.h5")

LABELS={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
# Set up drawing variables
drawing = False
brush_size = 5

# Fill the background with white
window.fill(white)

number_xcord=[]
number_ycord=[]

image_cnt=1
PREDICT = True

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Start drawing when mouse button is pressed
        if event.type == MOUSEBUTTONDOWN:
            drawing = True

        # Stop drawing when mouse button is released
        if event.type == MOUSEBUTTONUP:
            drawing = False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord) 

            rec_min_x,rec_max_x=max(number_xcord[0]-10,0),min(window_size[0],number_xcord[-1]+10)
            rec_min_y,rec_max_y=max(number_ycord[0]-10,0),min(window_size[1],number_ycord[-1]+10)

            number_xcord=[]
            number_ycord=[]

            img_arr=np.array(pygame.PixelArray(window))[rec_min_x:rec_max_x,rec_min_y:rec_max_y].T.astype(np.float32)
            
            pygame.draw.rect(window, red, (rec_min_x, rec_min_y, rec_max_x - rec_min_x, rec_max_y - rec_min_y), 2)

            
            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_cnt+=1
            
            if PREDICT:
              image=cv2.resize(img_arr,(28,28))
              image=np.pad(image,(10,10),'constant',constant_values=0)
              image=cv2.resize(image,(28,28))/255

              label=str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
              text_surface=FONT.render(label,True,red,white)
              textRecObj = text_surface.get_rect()
              textRecObj.left,textRecObj.bottom=rec_min_x,rec_max_y

              window.blit(text_surface,textRecObj)

        # Draw on the window
        if event.type == MOUSEMOTION:
            if drawing:
                mouse_x, mouse_y = event.pos
                pygame.draw.circle(window, black, (mouse_x, mouse_y), brush_size)
                number_xcord.append(mouse_x)
                number_ycord.append(mouse_y)

    # Update the display
    pygame.display.update()