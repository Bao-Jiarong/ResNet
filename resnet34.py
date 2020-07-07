''''
  Author       : Bao Jiarong
  Creation Date: 2020-07-07
  email        : bao.salirong@gmail.com
  Task         : ResNet34
 '''

import tensorflow as tf
import sys

class Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides = 1, is_begaining = False):
        super(Block, self).__init__()
        self.is_begaining = is_begaining
        self.convs = []
        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3),
                                                 strides = strides, activation  = "relu",
                                                 padding = "same"))

        self.convs.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3),
                                                 strides = (1,1), activation  = "linear",
                                                 padding = "same"))

        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1,1),strides = strides, activation = "linear",padding = "same")

    def call(self, inputs, **kwargs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        if self.is_begaining == True:
            inputs = self.conv2(inputs)
        x = x + inputs
        x = tf.keras.activations.relu(x)
        return x

#-------------------------------------------------------------------------------
class Resnet34(tf.keras.Model):
    def __init__(self, classes,filters = 64):
        super(Resnet34,self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (7,7),strides = (2,2), activation = "relu",padding = "same")
        self.pool1  = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.block1_1 = Block(filters)
        self.block1_2 = Block(filters)
        self.block1_3 = Block(filters)

        self.block2_1 = Block(filters << 1, 2, True)
        self.block2_2 = Block(filters << 1)
        self.block2_3 = Block(filters << 1)
        self.block2_4 = Block(filters << 1)

        self.block3_1 = Block(filters << 2, 2, True)
        self.block3_2 = Block(filters << 2)
        self.block3_3 = Block(filters << 2)
        self.block3_4 = Block(filters << 2)
        self.block3_5 = Block(filters << 2)
        self.block3_6 = Block(filters << 2)

        self.block4_1 = Block(filters << 3, 2, True)
        self.block4_2 = Block(filters << 3)
        self.block4_3 = Block(filters << 3)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc   = tf.keras.layers.Dense(units = classes, activation ="softmax")

    def call(self, inputs, training = None):
        # Backbone
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)
        x = self.block3_6(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)

        # Top
        x = self.pool(x)
        x = self.fc(x)
        return x

#------------------------------------------------------------------------------
def ResNet34(input_shape, classes, filters = 64):
    model = Resnet34(classes,filters)
    model.build(input_shape = input_shape)
    return model
