# coding:utf8
import cv2
def detect_video(video):    
    camera = cv2.VideoCapture(video)    
    history = 20      
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)    
    frames = 0    
    while True:        
        res, frame = camera.read()        
        if not res:            
            break        
        fg_mask = bs.apply(frame)   
        if frames < history:            
            frames += 1            
            continue        
        
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]        
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)        
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)        
        
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        for c in contours:            
            
            x, y, w, h = cv2.boundingRect(c)            
            
            area = cv2.contourArea(c)            
            if 500 < area < 3000:                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)        
        cv2.imshow("detection", frame)        
        cv2.imshow("back", dilated)       
        if cv2.waitKey(110) & 0xff == 27:            
            break    
    camera.release()

if __name__ == '__main__':
    video = './data/veh_640x480.mp4'    
    detect_video(video)