#!/usr/bin/env python
# coding: utf-8

import os, numpy as np, cv2

# input and output directory
inputdir='Inputs'
outputdir='Outputs'

if not os.path.isdir(inputdir):
  os.mkdir(inputdir)
if not os.path.isdir(outputdir):
  os.mkdir(outputdir)

model_path = 'yolov3.weights'
cfg_path = 'yolov3.cfg'
coconames_path = 'coco.names'
confidence_threshold=0.8
non_maximal_suppression=0.7

class YOLO:

  def __init__(self, model_path, cfg_path, coconames_path,
               confidence_threshold, non_maximal_suppression):
    
    self.model_path = model_path
    self.cfg_path = cfg_path
    self.coconames_path = coconames_path
    self.confidence_threshold = confidence_threshold
    self.non_maximal_suppression = non_maximal_suppression

    self.__net=cv2.dnn.readNet(self.model_path, self.cfg_path)
    print("Model weights and config have been loaded")

    with open(coconames_path) as f:
      self.classlabels=[line.strip() for line in f]
      print(f'Considering {len(self.classlabels)} classes.')


  def loadimage(self, imagefilepath):

    img = cv2.imread(imagefilepath)
    self.height, self.width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop=False)
    self.__net.setInput(blob)
    print(f'Loaded {imagefile} with height: {self.height}, width: {self.width}')
    return img


  def saveimage(self, img, outputimagefilepath):

    cv2.imwrite(outputimagefilepath, img)
    print(f'Saved results at {outputimagefilepath}')


  def detect_object(self, imagefilepath, outputdir, verbose=False):

    img=self.loadimage(imagefilepath)

    output_layers_names = self.__net.getUnconnectedOutLayersNames()
    layerOutputs = self.__net.forward(output_layers_names)

    boxes=[]
    confidences=[]
    class_ids=[]

    for output in layerOutputs:
      for detection in output:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]

        if confidence > self.confidence_threshold:
          x, y, w, h = detection[0]*self.width, detection[1]*self.height, detection[0]*self.width, detection[3]*self.height
          boxes.append([int(x), int(y), int(w), int(h)])
          confidences.append(float(confidence))
          class_ids.append(class_id)

    intermediate_results =list(zip(boxes, confidences, class_ids))
    indicies = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.non_maximal_suppression)
    results = np.array(intermediate_results)[indicies.flatten()].tolist()

    newimg=img.copy()

    font = cv2.FONT_HERSHEY_PLAIN
    colors=np.random.uniform(0, 255, size=(len(boxes), 3))

    for i, ((x, y, w, h), confidence, class_id) in enumerate(results):
      cv2.rectangle(newimg, (x-w//2, y-h//2), (x+w//2, y+h//2), colors[i], 2)
      cv2.putText(newimg, self.classlabels[i]+": "+"{:.2f}".format(confidence), (x-w//2, y+h//2-10), font, 1, colors[i], 2)

    self.saveimage(newimg, os.path.join(outputdir, f'detected_{imagefile}'))

if __name__ == "__main__":

  model=YOLO(model_path, cfg_path, coconames_path, confidence_threshold, non_maximal_suppression)
  for imagefile in os.listdir(inputdir):
    imagefilepath=os.path.join(inputdir, imagefile)
    model.detect_object(imagefilepath, outputdir)