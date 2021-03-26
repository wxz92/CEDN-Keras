import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.keras.utils import conv_utils
import numpy as np
import math
import cv2

# 池化
class MaxPoolingWithArgmax2D(Layer):
    '''MaxPooling for unpooling with indices.

    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)

    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

    def call(self, inputs, **kwargs):
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        padding = self.padding.upper()
        output, argmax = nn_ops.max_pool_with_argmax(inputs, ksize, strides, padding)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

# 反池化
class MaxUnpooling2D(Layer):
    '''Inversion of MaxPooling with indices.

    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)

    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''

    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]

        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (
            input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)

# 卷积+Relu
def Conv_Relu(BottomLayer, filters, kernel_size=(3,3), strides=(1,1), padding='same', dilation_rate=(1, 1), layerName=None):
    layer = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                   padding=padding, name=layerName, dilation_rate=dilation_rate)(BottomLayer)
    layer = tf.keras.layers.Activation('relu')(layer)
    return layer

# 创建网络
def Create_Model(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH):
    input = tf.keras.layers.Input(shape=(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 3), name='Input', batch_size=None)
    layerName = 'conv1_1'
    conv1_1 = Conv_Relu(input, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv1_2'
    conv1_2 = Conv_Relu(conv1_1, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'pool1'
    pool1, pool1_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2),
                                                 padding='valid', name=layerName)(conv1_2)
    layerName = 'conv2_1'
    conv2_1 = Conv_Relu(pool1, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv2_2'
    conv2_2 = Conv_Relu(conv2_1, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'pool2'
    pool2, pool2_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2),
                                                 padding='valid', name=layerName)(conv2_2)
    layerName = 'conv3_1'
    conv3_1 = Conv_Relu(pool2, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv3_2'
    conv3_2 = Conv_Relu(conv3_1, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv3_3'
    conv3_3 = Conv_Relu(conv3_2, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'pool3'
    pool3, pool3_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2),
                                                 padding='valid', name=layerName)(conv3_3)
    layerName = 'conv4_1'
    conv4_1 = Conv_Relu(pool3, 512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv4_2'
    conv4_2 = Conv_Relu(conv4_1, 512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'conv4_3'
    conv4_3 = Conv_Relu(conv4_2, 512, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              layerName=layerName)
    layerName = 'pool4'
    pool4, pool4_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2),
                                                 padding='same', name=layerName)(conv4_3)
    layerName = 'conv5_1'
    conv5_1 = Conv_Relu(pool4, 512, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', layerName=layerName)
    layerName = 'conv5_2'
    conv5_2 = Conv_Relu(conv5_1, 512, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', layerName=layerName)
    layerName = 'conv5_3'
    conv5_3 = Conv_Relu(conv5_2, 512, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', layerName=layerName)
    layerName = 'pool5'
    pool5, pool5_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2),
                                                 padding='same', name=layerName)(conv5_3)
    # ---------------------conv6---------------------------------
    layerName = 'conv6_1'
    conv6_1 = Conv_Relu(pool5, 4096, kernel_size=(7, 7), strides=(1, 1),
                              padding='same', layerName=layerName)
    conv6_1 = tf.keras.layers.Dropout(0.5)(conv6_1)
    layerName = 'conv6_2'
    conv6_2 = Conv_Relu(conv6_1, 512, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', layerName=layerName)
    conv6_2 = tf.keras.layers.Dropout(0.5)(conv6_2)

    # ---------------------unpool-deconv-5---------------------------------
    layerName = 'up5'
    up5 = MaxUnpooling2D(name=layerName)([conv6_2, pool5_argmax])
    _,_,_,c = conv6_2.get_shape()
    b,h,w,_ = conv5_3.get_shape()
    up5.set_shape([b,h,w,c])
    layerName = 'deconv5'
    deconv5 = Conv_Relu(up5, 512, kernel_size=(5, 5), strides=(1, 1),
                              padding='same', layerName=layerName)
    deconv5 = tf.keras.layers.Dropout(0.5)(deconv5)
    # ---------------------unpool-deconv-4---------------------------------
    layerName = 'up4'
    up4 = MaxUnpooling2D(name=layerName)([deconv5, pool4_argmax])
    _, _, _, c = deconv5.get_shape()
    b, h, w, _ = conv4_3.get_shape()
    up4.set_shape([b, h, w, c])
    layerName = 'deconv4'
    deconv4 = Conv_Relu(up4, 256, kernel_size=(5, 5), strides=(1, 1),
                              padding='same', layerName=layerName)
    deconv4 = tf.keras.layers.Dropout(0.5)(deconv4)
    # ---------------------unpool-deconv-3---------------------------------
    layerName = 'up3'
    up3 = MaxUnpooling2D(name=layerName)([deconv4, pool3_argmax])
    _, _, _, c = deconv4.get_shape()
    b, h, w, _ = conv3_3.get_shape()
    up3.set_shape([b, h, w, c])
    layerName = 'deconv3'
    deconv3 = Conv_Relu(up3, 128, kernel_size=(5, 5), strides=(1, 1),
                              padding='same', layerName=layerName)
    deconv3 = tf.keras.layers.Dropout(0.5)(deconv3)
    # ---------------------unpool-deconv-2---------------------------------
    layerName = 'up2'
    up2 = MaxUnpooling2D(name=layerName)([deconv3, pool2_argmax])
    _, _, _, c = deconv3.get_shape()
    b, h, w, _ = conv2_2.get_shape()
    up2.set_shape([b, h, w, c])
    layerName = 'deconv2'
    deconv2 = Conv_Relu(up2, 64, kernel_size=(5, 5), strides=(1, 1),
                              padding='same', layerName=layerName)
    deconv2 = tf.keras.layers.Dropout(0.5)(deconv2)
    # ---------------------unpool-deconv-1---------------------------------
    layerName = 'up1'
    up1 = MaxUnpooling2D(name=layerName)([deconv2, pool1_argmax])
    _, _, _, c = deconv2.get_shape()
    b, h, w, _ = conv1_2.get_shape()
    up1.set_shape([b, h, w, c])
    layerName = 'deconv1'
    deconv1 = Conv_Relu(up1, 32, kernel_size=(5, 5), strides=(1, 1),
                              padding='same', layerName=layerName)
    deconv1 = tf.keras.layers.Dropout(0.5)(deconv1)
    # ---------------------pred1-contour---------------------------------
    layerName = 'pred1-contour'
    pred1_contour = tf.keras.layers.Conv2D(1, kernel_size=(5, 5), strides=(1, 1),
                                           padding='same', name=layerName)(deconv1)
    pred1_contour = tf.keras.layers.Activation(activation='sigmoid')(pred1_contour)

    """ END """
    model = Model(inputs=input, outputs=pred1_contour)
    model.summary()
    return model

# 加载模型
model = Create_Model(224, 224)
model.load_weights('model.h5')
# 读图
picPath = '000999.jpg'
image = cv2.imdecode(np.fromfile(str(picPath), dtype=np.uint8), -1)
# 图像宽高必须是32的整数倍，否则需要扩充边界
h = image.shape[0]
w = image.shape[1]
hExpand = math.ceil(h / 32)*32
wExpand = math.ceil(w / 32)*32
top = math.floor((hExpand-h)/2)
bottom = hExpand-h - top
left = math.floor((wExpand-w)/2)
right = wExpand-w-left
image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_REPLICATE)
# 识别
image = tf.cast(image, tf.float32)
image = tf.expand_dims(image, axis=0)
result = model.predict({'Input': image})
result = np.squeeze(result)
result = result[top:top+h, left:left+w]
# sigmoid结果范围[0,1]，阈值取0.5
result[result > 0.5] = 1
result[result <= 0.5] = 0
# 显示
result = result*255
result = result.astype(np.uint8)
cv2.imshow("Result", result)
cv2.waitKey()