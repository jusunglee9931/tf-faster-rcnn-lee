from PIL import Image,ImageDraw,ImageFont
import random

colorlist = ['white', 'red', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange']
fontname  = "NotoSansKR-Black.otf"
iou_thresh = 0.3
colorlist = ['white', 'red', 'blue', 'green', 'yellow', 'brown', 'purple', 'orange']
def singekorfunc():
  start = 0xB0A1
  u_idx = (start >> 8) & 0xff
  l_idx = start & 0xff
  list = ['back']
  i = 0
  while 1 :
   u_idx_s = chr(u_idx)
   l_idx_s = chr(l_idx)
   s = u_idx_s+l_idx_s
   #print (s.decode('EUC-KR').encode('UTF-8'))
   list.append(s)
   i +=1
   if u_idx == 0xc8 and l_idx == 0xfe :
       break
   if l_idx == 0xfe:
       l_idx = 0xa1
       u_idx += 1
   else :
       l_idx += 1

  #print("i :%d" %i)
  return list
def addaddtionalword(list):
    '''
    code = [0xA3B0, 0xA3B1, 0xA3B2, 0xA3B3, 0xA3B4, 0xA3B5, 0xA3B6, 0xA3B7, 0xA3B8,0xA3B9,0xA3A5, 0xA3A8, 0xA3A9,
            0xA3C1, 0xA3C2, 0xA3C3, 0xA3C4, 0xA3C5, 0xA3C6, 0xA3C7, 0xA3C8, 0xA3C9, 0xA3CA, 0xA3CB, 0xA3CC, 0xA3CD,
            0xA3CE, 0xA3CF,0xA3D0, 0xA3D1, 0xA3D2, 0xA3D3, 0xA3D4, 0xA3D5, 0xA3D6, 0xA3D7, 0xA3D8, 0xA3D9, 0xA3DA,
            0xA3E1, 0xA3E2, 0xA3E3, 0xA3E4, 0xA3E5, 0xA3E6, 0xA3E7, 0xA3E8, 0xA3E9, 0xA3EA, 0xA3EB, 0xA3EC, 0xA3ED,
            0xA3EE, 0xA3EF, 0xA3F0, 0xA3F1, 0xA3F2, 0xA3F3, 0xA3F4, 0xA3F5, 0xA3F6, 0xA3F7, 0xA3F8, 0xA3F9, 0xA3FA,
            ]
    for byte in code:
        u_idx = (byte >> 8) & 0xff
        l_idx = byte & 0xff
        u_idx_s = chr(u_idx)
        l_idx_s = chr(l_idx)
        s = u_idx_s + l_idx_s
        list.append(s)
        print(s.decode('EUC-KR').encode('UTF-8'))
    '''
    code = ['dummy1','0','1','2','3','4','5','6','7','8','9','%','(',')','~','dummy2',
            'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
            'A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O' ,'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z'
            ]
    for byte in code:
        list.append(byte)


    return list


def caliou(boxa,boxb):
    xA = max(boxa[0],boxb[0])
    yA = max(boxa[1],boxb[1])
    xB = min(boxa[2],boxb[2])
    yB = min(boxa[3],boxb[3])
    intersection  = (xB - xA + 1)*(yB - yA +1)
    area_boxa = (boxa[2] - boxa[0]+1)*(boxa[3] - boxa[1] + 1)
    area_boxb = (boxb[2] - boxb[0] + 1) * (boxb[3] - boxb[1] + 1)

    return intersection/(area_boxa + area_boxb - intersection)



class rowinfo(object):
    def __init__(self , center, height):
        self.center = center
        self.height = height
        self.n = 1
        self.boundbox_list = []
        self.wordlist = singekorfunc()
        self.wordlist = addaddtionalword(self.wordlist)
    def getuplow(self):
        return (self.center + self.height/2),(self.center - self.height/2)
    def update(self,box):
        n = self.n
        b_center = (box[1]+box[3])/2
        #print("b_center:%d"%b_center)
        height = self.height
        self.center = (n*self.center + b_center)/(n+1)
        self.height = 2* max(height/2, max(box[3] - self.center,self.center - box[1]))
        #print("updated value :%f %f" % (self.center, self.height))
        self.n += 1
    def addbox(self, box):
        self.boundbox_list.append(box)

    def sortbox(self):
        self.boundbox_list = sorted(self.boundbox_list,key = lambda bound:bound[0])

    def __repr__(self):
        return repr((self.center,self.height))
    def getboxes(self):
        return self.boundbox_list
    def getnumclass(self):
        return len(self.wordlist)
    def getword(self,idx):
        return self.wordlist[idx]
    def filterbox(self):
        del_list = {}
        length = len(self.boundbox_list)
        for i in range(0,length):
            for j in range(i+1,length):
                if del_list.get(i) == 1 or del_list.get(j) == 1 :
                    continue
                iou = caliou(self.boundbox_list[i],self.boundbox_list[j])
                if iou > iou_thresh :
                    if self.boundbox_list[i][4] > self.boundbox_list[j][4]:
                        del_list[j] = 1
                    else:
                        del_list[i] = 1
        re_del_list = del_list.keys()
        re_del_list.sort()
        re_del_list.reverse()
        #print(re_del_list)
        for idx in re_del_list:

            del self.boundbox_list[idx]


    def getstring(self):
        txt = u""
        lens = len(self.boundbox_list)
        for i,boxes in enumerate(self.boundbox_list):
            ch =self.wordlist[int(boxes[5])]
            txt +=ch.decode('EUC-KR')
            if i + 1 < lens:
                next_boxes = self.boundbox_list[i+1]
                if next_boxes[0] - boxes[2] > 0.01 * self.height:
                    txt +=u" "

        #print(txt)
        return txt



class line_recog(object):
    def __init__(self,box,img,im_name):
        self.box = box
        self.img = img
        self.rowlist = []
        self.im_name = im_name
        self.row_number = 0;


    def recognize(self):
        if len(self.box) == 0:
            return
        self.__makerow(self.box[0])
        for box in self.box:
            #print(box)
            self.__contain_line(box)
        #debug
        self.rowlist= sorted(self.rowlist,key=lambda rowinfo:rowinfo.center)
        for row in self.rowlist:
            row.filterbox()
            row.sortbox()





    def draw(self):
        img = self.img
        w,h = img.size
        brush = ImageDraw.Draw(img)
        for row in self.rowlist:
            c_idx = random.randrange(0,len(colorlist))
            up,low = row.getuplow()
            brush.line([0,up,w-1,up], fill = colorlist[c_idx])
            brush.line([0,low,w-1,low], fill = colorlist[c_idx])
        #print(self.im_name)
        img.save(self.im_name+"_line.jpg")
    def writetxt(self):
        f =open(self.im_name +".txt",mode = 'w')
        for row in self.rowlist:
            txt = row.getstring()
            f.write(txt.encode("UTF-8"))
            f.write(u"\n".encode("UTF-8"))
        f.close()
    def gettxt(self):
        txt_string = u""
        for row in self.rowlist:
            txt = row.getstring()
            txt_string += txt +u"\n"
        return txt_string.encode("UTF-8")
    def getboximg(self,factor):
        img = self.img
        brush = ImageDraw.Draw(img)
        font = ImageFont.truetype(fontname, 20)
        for row in self.rowlist:
            boxes = row.getboxes()
            for box in boxes:
                box[0] = box[0] / factor
                box[1] = box[1] / factor
                box[2] = box[2] / factor
                box[3] = box[3] / factor
                fontsize = int((box[2]-box[0])*0.5)
                font = ImageFont.truetype(fontname,fontsize )
                c_idx = random.randrange(0, len(colorlist))
                brush.rectangle([(box[0], box[1]), (box[2], box[3])],font=font, outline=colorlist[c_idx])
                text = row.getword(int(box[5])).decode('EUC-KR')
                #string = u'{0} {1:.2f}'.format(text, box[4])
                #brush.text((box[0], box[1] - fontsize), string, font=font, fill=colorlist[c_idx])
        #print(self.im_name)
        #img.save(self.im_name+"_line.jpg")
        return img
    def saveimg(self,factor):
        img = self.getboximg(factor)
        img.save(self.im_name +".jpg")

    def __contain_line(self,box):
        length = len(self.rowlist)
        ck = 0
        for i in range(0,length):
            row = self.rowlist[i]
            if self.__compare(row,box) :
                #print("row is updated")
                row.update(box)
                ck += 1
                break

        if ck == 0 :
         self.__makerow(box)


    def __compare(self,row,box):
        #r_up,r_low = row.getuplow()
        r_center = row.center

        if box[1] < r_center and box[3] > r_center :
            row.addbox(box.tolist())
            return 1

        #print("box value :%f %f r_center : %f " %(box[1], box[3], r_center))
        return 0




    def __makerow(self,box):
        center = (box[1]+box[3])/2
        height = 2*(box[3] - center)
        rows = rowinfo(center,height)
        rows.addbox(box.tolist())
        self.rowlist.append(rows)
        self.row_number +=1
        #print("row is created %d %f %f" %(self.row_number,center,height))








