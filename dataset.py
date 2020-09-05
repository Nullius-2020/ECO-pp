import paddle.fluid.dataloader as data
import paddle
from PIL import Image
import os
import os.path
import numpy as np
import random
from numpy.random import randint
from opts import parser

args = parser.parse_args()

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(object):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.batch_size=args.batch_size
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()
        

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        #print(record.num_frames > self.num_segments + self.new_length - 1)

        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
            #print(offsets)
        #print(offsets+1)
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self,index):

        batch = 0
        imgs=[]
        labels=[]
        i=index*self.batch_size

        if self.random_shift:
           random.shuffle(self.video_list)
        while i < (len(self.video_list)):
            record = self.video_list[i]
            if not self.test_mode:
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            else:
                segment_indices = self._get_test_indices(record)  

            img, label = self.get(record, segment_indices)
           
            img=np.array(img).astype('float32')
            label=np.array(label).astype('int64')
            
            imgs.append(img)
            labels.append(label)
            batch += 1
            i+=1

            if batch == self.batch_size:

                bimgs=np.array(imgs).reshape(-1,3,224,224)
                blabels=np.array(labels).reshape(-1,1)
                break
            
        if batch == self.batch_size:
            return  bimgs,blabels
      

        

    def get(self, record, indices):

        images = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
    


if __name__ == '__main__':

    from transforms import *
    from models import *
    import paddle.fluid as fluid
    
    fset = TSNDataSet("", 'data/ucf101_rgb_train_t.txt', num_segments=24,
                   new_length=1,
                   modality='RGB',
                   random_shift=False,
                   test_mode=False,
                   image_tmpl='img_'+'{:05d}.jpg' if args.modality in ["RGB", "RGBDiff"] else 'img_'+'{:05d}.jpg',
                   transform=Compose([
                       GroupScale(int(224)),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       IdentityTransform(),
                   ]))
    def batch_generator_creator():
        def __reader__():
                batch =0
                img=[]
                batch_data=[]
                label=[]
                for i in range(len(fset)):
                    record = fset.video_list[i]
                    if not fset.test_mode:
                        segment_indices = fset._sample_indices(record) if fset.random_shift else fset._get_val_indices(record)
                    else:
                        segment_indices = fset._get_test_indices(record)  
                    print(record.path)
                    img, label = fset.get(record, segment_indices)
           
                    img=np.array(img).astype('float32')
                    label=np.array(label).astype('int64')
                    batch_data.append([img,label])
                    batch += 1
                    if batch == fset.batch_size:
                        yield batch_data
                        batch =0
                        img=[]
                        batch_data=[]
                        label=[]
        return __reader__
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        t_loader= fluid.io.DataLoader.from_generator(capacity=10,return_list=True, iterable=True, drop_last=True)
        t_loader.set_sample_list_generator(batch_generator_creator(), places=place)
        #i=0
       
        batch=len(fset)//fset.batch_size
        for i in range (batch):
            
            print(i)
            img,lab =fset.__getitem__(i)
            img = np.array(img).astype('float32').reshape(-1,3,224,224)
            lab = np.array(lab).astype('int64').reshape(-1,1)
            #print('\n 8888888888888888888888888')
            if i==1:
                break
        i=0
        for image, label in t_loader():
            print(i)
            i+=1
            image=paddle.fluid.layers.reshape(image, shape=[-1,3,224,224])
            label=paddle.fluid.layers.reshape(label, shape=[-1,1])
            #print(i,image.shape,img.shape,label,lab)
            if i==2:
                break
       


        
