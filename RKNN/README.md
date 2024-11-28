# Mobile_Face_Process_RK3588

RK3588 上部署rknn的人脸检测、关键点、识别模型相关模型

下载完成后，将模型放置到 assert_rknn/ 文件夹下
~~~
assert 
├── face_detector.rknn
├── face_landmarks.rknn
├── facenet128.rknn
├── facial_expression.rknn
└── model.txt

~~~

人脸数据库在face_dataset/文件夹下

## Running
### 在电脑上基于rknpu进行测试

单张图像测试
~~~
python main_img.py
~~~

视频测试 facedb重新加载人脸库
~~~
python main_rknn.py --video demo.mov --facedb
~~~

摄像头测试 这里我用的双目摄像头，会采集两个图像，且默认cam id = 20, 正常应该为0
~~~
python main_rknn.py
~~~

### 在rk3588开发板上进行测试

~~~
# 在电脑上基于 rknpu 测试，使用的是from rknn.api import RKNN as RKNNLite
# 切换到rk3588上，需要 使用 from rknnlite.api import RKNNLite

from rknnlite.api import RKNNLite
#from rknn.api import RKNN as RKNNLite
~~~







## Citation
[deepface](https://github.com/serengil/deepface)
[head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
[Peppa_Pig_Face_Landmark](https://github.com/610265158/Peppa_Pig_Face_Landmark)

