# Code : semi-auto balancing data（based on Albumentations）



* The library is faster than other libraries on most of the transformations.
* Based on numpy, OpenCV, imgaug (directly based on Albumentations) picking the best from each of them.

## Installation
You can first use pip to install albumentations:
```
pip install albumentations
```


## How to use(more details could be shown in the notebook)
Taking the kaggle competition APTOS 2019 Blindness Detection as an example。

### Loading picture data and csv data,as well as visualzing how unbalance the data is.
```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_3 = pd.read_csv("your_csv_path") #to load the data csv 
f,ax=plt.subplots(1,2,figsize=(18,8))
df_3['diagnosis'].value_counts().plot.pie(explode=[0,0,0,0,0],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('diagnosis')
ax[0].set_ylabel('')
sns.countplot('diagnosis',data=df_3,ax=ax[1])
ax[1].set_title('diagnosis')
plt.show()
```
### calculate the number of data needed to add in
```python
#numbers of needing to add aug pics for label 1
aug_nums_1 = df_3.diagnosis.value_counts().max()-df_3.diagnosis.value_counts()[1]
#numbers of needing to add aug pics for label 2
aug_nums_2 = df_3.diagnosis.value_counts().max()-df_3.diagnosis.value_counts()[2]
#numbers of needing to add aug pics for label 3
aug_nums_3 = df_3.diagnosis.value_counts().max()-df_3.diagnosis.value_counts()[3]
#numbers of needing to add aug pics for label 4
aug_nums_4 = df_3.diagnosis.value_counts().max()-df_3.diagnosis.value_counts()[4]
print(aug_nums_1,aug_nums_2,aug_nums_3,aug_nums_4)
```
### define some functions to be used in data augmentation
```python
from urllib.request import urlopen

import numpy as np
import cv2
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,Rotate,RGBShift,ChannelShuffle,GaussNoise,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,RandomBrightness,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,GaussianBlur,RandomBrightnessContrast,RandomSnow
)

#image = cv2.imread("/home/noctis/下载/#APTOS/concat_data/2019+IDRID/0a74c92e287c.png")#download_image('https://d177hi9zlsijyy.cloudfront.net/wp-content/uploads/sites/2/2018/05/11202041/180511105900-atlas-boston-dynamics-robot-running-super-tease.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def strong_aug(p=0.9):
    return Compose([
        Rotate(limit=45,border_mode=1),
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.4),
        OneOf([
            MotionBlur(p=1),
            MedianBlur(p=1),
            Blur(p=1),
            GaussianBlur(p=1)
        ], p=0.4),
        
        OneOf([
            CLAHE(clip_limit=2,p=1),
            IAASharpen(p=1),
            IAAEmboss(p=1),
            RandomBrightnessContrast(p=1),            
        ], p=0.3),
    ], p=p)


```
###  Taking label 1 as an example, showing how to extend data on local mechine
```
loading_path = "/home/noctis/下载/#APTOS/concat_data/2019+IDRID/"
writing_path = "/devdata/#APTOS/balanced-data/"
aug_df = pd.DataFrame(columns=['id_code','diagnosis'])


df_label1_add = df_3[df_3["diagnosis"]==1].sample(n=aug_nums_1,replace=True,random_state=1)

for count,ids in enumerate(df_label1_add["id_code"]):
    read_pic_path = loading_path + ids + ".png"
    write_name = ids + "aug{}".format(count)
    write_pic_path = writing_path + write_name + ".png"
    image = cv2.imread(read_pic_path)
    aug = strong_aug(p=0.9)
    image = aug(image=image)['image']
    aug_df = aug_df.append({'id_code': write_name,'diagnosis':1}, ignore_index=True)
    cv2.imwrite(write_pic_path, image)
    
    print(count,ids)

```



# Citing（Thanks for the Albumentations）
If you find this library useful for your research, please consider citing:(and star if u like this notebook)

```
@article{2018arXiv180906839B,
    author = {A. Buslaev, A. Parinov, E. Khvedchenya, V.~I. Iglovikov and A.~A. Kalinin},
     title = "{Albumentations: fast and flexible image augmentations}",
   journal = {ArXiv e-prints},
    eprint = {1809.06839},
      year = 2018      
}
```
