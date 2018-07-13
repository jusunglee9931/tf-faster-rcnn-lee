import numpy as np
import os
import random

from PIL import Image, ImageDraw

MAX_SIZE = 1000.0
class dataloader(object):
    def __init__(self,path,gt_path):
        self.path    = path
        self.gt_path = gt_path
        self.getlist()
        self.cnt = 0


    def getlist(self):
        self.list = [f for f in os.listdir(self.path)]
        self.lens = len(self.list)

    def getgt(self,index,ratio):
        filepath = self.list[index].split(".")[0]
        filepath = os.path.join(self.gt_path,"gt_"+filepath+".txt")
        gt_boxes = []
        #print(filepath)
        f = open(filepath)
        while True :
            buffer = f.readline()
            if not buffer:
                break
            buffer = buffer.split(" ")
            boxes = [int(buffer[0])/ratio,int(buffer[1])/ratio,int(buffer[2])/ratio,int(buffer[3])/ratio,int(1)]
            gt_boxes.append(boxes)
        return np.asarray(gt_boxes).reshape((-1,5))



    def getimage(self,index):
        filepath = os.path.join(self.path, self.list[index])
        im = Image.open(filepath)
        width, height = im.size
        ratio = 1
        if width > MAX_SIZE or height > MAX_SIZE:
            ratio = max(width,height)/MAX_SIZE
            im = im.resize((int(width/ratio),int(height/ratio)))


        return np.asarray(im,dtype=np.float32) / 255 ,ratio,im

    def fetch(self):
        if self.cnt == self.lens:
            self.cnt = 0

        blob={}
        img, ratio,blob['pil_im'] = self.getimage(self.cnt)
        blob['data'] = np.expand_dims(img, axis=0)
        blob['gt_boxes'] = self.getgt(self.cnt,ratio)


        #print(blob['data'].shape)
        #print(blob['gt_boxes'].shape)
        blob['im_info'] =np.array([blob['data'].shape[1], blob['data'].shape[2], 1], dtype=np.float32)
        self.cnt += 1
        return blob




if __name__ == "__main__":
    loader = dataloader("/data/Challenge2_Training_Task12_Images","/data/Challenge2_Training_Task1_GT")
    loader.fetch()