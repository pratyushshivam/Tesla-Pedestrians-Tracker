import cv2

#Our Image
img_file='CarImage3.jpg'

#video = cv2.VideoCapture('')
video = cv2.VideoCapture('video1.mp4')
video1 = cv2.VideoCapture('ped.mp4')


#Our pre-trained car classifier
classifier_file = 'car_detector.xml'


# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file) #brain
pedestrian_tracker = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

#run forever until car stops or crashes
while True:
    #read the current frame
    read_succesful,frame = video1.read() # reads one frame of video
    #read_successful says if the read was successful or not, and frame is what we are interested in, and they are a tuple
    
    
    
    #safe coding
    if read_succesful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break  

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame) 
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame) 


    # Draw rectangles around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Draw rectangles around the pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)    



    # Display the image with the cars spotted
    cv2.imshow('video',frame) #pops a window with the image 

    #Don't autoclose
    cv2.waitKey(1) # waits for a key to be pressed to close window










"""













#create opencv image
img = cv2.imread(img_file)

#convert to grayscale (needed for haarcascade)
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#detect cars
cars = car_tracker.detectMultiScale(black_n_white)  #multiscale means detect cars of any scale


# print(cars)

# output [top,left,width,height]
# [[178 720  54  54]
#  [814 294  73  73]
#  [881 759  94  94]
#  [190 733  34  34]]
#[x,y,w,h]


# Draw rectangles around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,40,255),2)








# Display the image with the cars spotted
cv2.imshow('Car_detector',img) #pops a window with the image 

#Don't autoclose
cv2.waitKey() # waits for a key to be pressed to close window

"""


print("cc")