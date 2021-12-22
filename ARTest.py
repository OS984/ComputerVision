'''
->We need to have some abstract loop object implementing the 'while True' segment; this can be the abstraction of our webcam

->Frame object inside of the Webcam

->Frame manipulator class (not instantiable!) implementing methods for playing with the frame.

->Prop object holding animations/original orientations etc

'''



#<IMPORTS
import face_recognition as frec
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#IMPORTS/>



#<REFERENCE
class FaceReference:
    
    def __init__(self, points : np.ndarray, size : tuple, loc : tuple, marks : dict):
        self.size = size
        self.loc = loc
        self.marks = marks
        self.points = points
        
    def getLoc(self) -> tuple:
        return self.loc
    
    def getSize(self) -> tuple:
        return self.size
    
    def getLandmarks(self) -> dict:
        return self.marks
    
    def getPoints(self) -> np.ndarray:
        return self.points
#REFERENCE/>



#<EYETRACKER
class EyeTracker:
    
    def __init__(self, blink_threshold : float):
        self.blink_thresh = blink_threshold
        
    def getEyestate(self, reference : FaceReference) -> float:
        left_marks, right_marks = (reference.getLandmarks()['left_eye'], reference.getLandmarks()['right_eye'])
        hor_metric = (left_marks[3][0] - left_marks[0][0], right_marks[3][0] - right_marks[0][0])
        ver_metric = (left_marks[4][1] - left_marks[1][1], right_marks[4][1] - right_marks[1][1])
        left_metric = ver_metric[0] / hor_metric[0]
        right_metric = ver_metric[1] / hor_metric[1]
        overall_metric = (right_metric + left_metric) / 2
        return overall_metric
#EYETRACKER/>



#<METRICPLOTTER
class MetricPlotter:
    
    def __init__(self):
        self.metric = np.array([])
        self.rolling = np.array([])
        
    def addData(self, val) -> None:
        self.metric = np.append(self.metric, val)
        
    def getData(self):
        return (np.array(range(len(self.metric))), self.metric)
    
    def getMean(self):
        return sum(self.metric) / len(self.metric)
    
    def getStd(self):
        return np.std(self.metric)
    
    def genRollingAverage(self, gap):
        #Take average eye value over some gap (determined from frame rate)
        i = 0
        while (i + gap) <= (len(self.metric) - 1):
            chunk = self.metric[i*gap:(i+1)*gap]
            self.rolling = np.append(self.rolling, np.mean(chunk))
            i += 1
        
    def getRollingData(self):
        return (np.array(range(len(self.rolling))), self.rolling)
    
#METRICPLOTTER/>



#<FRAME
class Frame:
    
    def __init__(self, data : np.ndarray):
        self.data = data
        self.im = Image.fromarray(data)

    def update(self, data : np.ndarray) -> None:
        self.data = data
        self.im = Image.fromarray(self.data)
        
    def getData(self) -> np.ndarray:
        return self.data
    
    def getImage(self) -> Image:
        return self.im
#FRAME/>



#<PROP
class Prop:
    
    def __init__(self, im_string : str, reference : FaceReference):
        self.reference = reference
        self.im = Image.open(im_string).resize(reference.size)
        self.data = np.array(self.im)
        
    def getImage(self) -> Image:
        return self.im
    
    def getReference(self) -> FaceReference:
        return self.reference
#PROP/>



#<FrameTools
class FrameTools:
    
    def getFaceReference(frame : Frame, x_corr : int, y_corr : int) -> FaceReference:
        #Get facial landmarks from face_recognition
        landmarks = frec.api.face_landmarks(frame.getData())[0]
        
        #Generate the corners of the box (Max we need is 3 for Affine transforms)
        box_tL = ((2*landmarks['left_eye'][0][0]) - landmarks['nose_tip'][0][0] - x_corr, (2*landmarks['left_eye'][0][1]) - landmarks['nose_tip'][-1][1] - y_corr)
        box_bR = ((2*landmarks['right_eye'][-1][0]) - landmarks['nose_tip'][0][0] + x_corr, (2*landmarks['nose_tip'][0][1]) - landmarks['right_eye'][-1][1] + y_corr)     
        box_bL = ((2*landmarks['left_eye'][0][0]) - landmarks['nose_tip'][0][0] - x_corr, (2*landmarks['nose_tip'][-1][1]) - landmarks['left_eye'][0][1] + y_corr)
        
        #Calculate reference size using box points
        loc = box_tL
        size = (box_bR[0] - box_tL[0], box_bL[1] - box_tL[1])
        
        #Generate unscaled points for use with affine transform (Box points are scaled so can't use without fucking things up)
        points = np.float32([landmarks['left_eye'][0], landmarks['right_eye'][-1], landmarks['nose_tip'][0]])
        
        return FaceReference(points, size, loc, landmarks)
    
    def getComposite(source_frame : Frame, prop : Prop) -> Image:
        clean = Image.fromarray(np.zeros_like(source_frame.getData()))
        clean.paste(prop.getImage(), prop.getReference().getLoc())
        return clean
        
    def getAffineTransform(source_ref : FaceReference, target_ref : FaceReference, frame_data : np.ndarray) -> Image:
        #Generate transformation
        transform = cv2.getAffineTransform(source_ref.getPoints(), target_ref.getPoints())
        
        #Do the affine transform
        affine_transformed = cv2.warpAffine(frame_data, transform, frame_data.shape[:2][::-1]) #discrepancy between how arrays think of shapes and cv2
        
        #Transform black pixels to alpha
        alpha = Image.fromarray(affine_transformed).convert("RGBA") #RGBA for ease of pasting
        alpha_data = np.array(alpha)
        r, g, b, a = alpha_data.T
        black_pixels = (r == 0) & (g == 0) & (b == 0) 
        alpha_data[...,:][black_pixels.T] = (0, 0, 0, 0)
        
        return Image.fromarray(alpha_data)
#FrameTools/>



#<FEED
class Feed:
    
    def __init__(self, prop_name : str, encoded=None):
        self.init_prop = None
        self.init_frame = None
        self.prop_name = prop_name
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(5)
        self.encoded = encoded #Placeholder for only mapping to my face eventually.
        self.mask_bool = False #Whether to apply the mask
        
    def mainLoop(self) -> None:
        #Do an inital read to get base positioning for prop before affine transform
        _, data = self.cap.read()
        data = cv2.resize(data, None, fx=0.5, fy=0.5) #Resize to keep decent performance
        self.init_frame = Frame(data)
        self.init_prop = Prop(self.prop_name, FrameTools.getFaceReference(self.init_frame, 5, 10))
        
        #Create dataplotter for eye movement
        plotter = MetricPlotter()
        
        #Create prop composite for later affine map
        composite = FrameTools.getComposite(self.init_frame, self.init_prop)
        
        #Actual mainloop where map is calculated and applied
        while True:
            #Try catch to get around failed facematching due to angle
            try:
                #Check key exit
                c = cv2.waitKey(1)
                if c == 27: #Escape key
                    break
                if c == 109: #m key
                    self.mask_bool = not self.mask_bool
                
                #Read the camera and generate frame
                _, data = self.cap.read()
                data = cv2.resize(data, None, fx=0.5, fy=0.5) #Resize to keep decent performance
                working_frame = Frame(data)
                working_face_reference = FrameTools.getFaceReference(working_frame, 5, 10)
                
                #Return some eye metrics
                tracker = EyeTracker(0.15)
                plotter.addData(tracker.getEyestate(working_face_reference))
                
                #Check whether we want the mask to be applied or not
                if self.mask_bool == True:
                    
                    #Generate and apply affine map
                    affined = FrameTools.getAffineTransform(self.init_prop.getReference(), working_face_reference, np.array(composite))
                
                    #Combine affined prop overlay with frame
                    frame_repr = working_frame.getImage()
                    frame_repr.paste(affined, (0,0), affined)
                    cv2.imshow('Masked cam', np.array(frame_repr))
                
                else:
                    
                    #Only show the bare webcam image
                    cv2.imshow('Masked cam', working_frame.getData())
            except Exception as err:
                print(err)
        
        #Clean up after cv2
        self.cap.release()
        cv2.destroyAllWindows()
        
        #Display eye data
        data = plotter.getData()
        plotter.genRollingAverage(int(self.fps // 2))
        rolling_data = plotter.getRollingData()
        plt.plot(rolling_data[0], rolling_data[1])
        plt.show()
#FEED/>

#<Mainloop
if __name__ == '__main__':
    feed = Feed('doc.png')
    feed.mainLoop()
#Mainloop/>
