# CEDN
CEDN原文献（caffe版本）的代码地址：https://github.com/jimeiyang/objectContourDetector

# 网络模型
根据原作者代码中的“vgg-16-encoder-decoder-contour.prototxt”重新搭建keras网络结构。

# 网络参数
根据作者提供的“vgg-16-encoder-decoder-contour-w10-pascal-iter030.caffemodel”，将卷积层参数读取出来，并加载到Keras卷积层中。

# 注意
只有运行代码，无训练代码。
