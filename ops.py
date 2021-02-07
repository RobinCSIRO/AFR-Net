import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib as tf_contrib

##################################################################################
# New Conv/Other Blocks Definition Based on Taki02_Github Design
##################################################################################
# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)


        return x

def dila_conv(x, channels, dr=2, use_bias=True, scope='dila_conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=3, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=1, use_bias=use_bias, dilation_rate=dr)


        return x

def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                       strides=stride, padding='SAME', use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def global_max_pooling(x):
    gmp = tf.reduce_max(x, axis=[1, 2])
    return gmp

def lrelu(x, alpha=0.1):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,epsilon=1e-05,center=True, scale=True,scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm
##################################################################################
# Refienement Blocks Design
##################################################################################
def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, scope='conv_1')
            #x = instance_norm(x)
            x = lrelu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, scope='conv_2')
            #x = instance_norm(x)

        return x + x_init

def RES_BLOCK(x, Block_Num=3, scope="RES_BLOCK"):
    with tf.variable_scope(scope):
        _,_,_,Cha = x.get_shape()
        for i in range(Block_Num):
            x = resblock(x, Cha, scope=scope+str(i+1))
        return x

def LEFT_REF_BLOCK(EN_0, EN_1, EN_2, EN_3, scope="LRB"):
    
    with tf.variable_scope(scope):
        DE_3 = RES_BLOCK(EN_3, Block_Num=3, scope="RES_BLOCK1") #90*120*256
        #print(DE_3.shape)

        _,_,_,Cha1 = EN_2.get_shape()
        DE_3_UP = deconv(DE_3, Cha1, kernel=3, stride=2, use_bias=True, scope='deconv_1')
        #DE_3_UP = instance_norm(DE_3_UP, scope='ins1')
        DE_3_UP = lrelu(DE_3_UP, alpha=0.1)
        #print(DE_1.shape)
        #print(DE_1_UP.shape)
        EN_2_POST = RES_BLOCK(EN_2, Block_Num=1, scope="RES_BLOCK2")
        DE_2_SKIP = DE_3_UP + EN_2_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        DE_2_SKIP_POST = RES_BLOCK(DE_2_SKIP, Block_Num=1, scope="RES_BLOCK3")

        _,_,_,Cha2 = EN_1.get_shape()
        DE_2_UP = deconv(DE_2_SKIP_POST, Cha2, kernel=3, stride=2, use_bias=True, scope='deconv_2')
        #DE_2_UP = instance_norm(DE_2_UP, scope='ins2')
        DE_2_UP = lrelu(DE_2_UP, alpha=0.1)
        #print(DE_1.shape)
        #print(DE_1_UP.shape)
        EN_1_POST = RES_BLOCK(EN_1, Block_Num=1, scope="RES_BLOCK4")
        DE_1_SKIP = DE_2_UP + EN_1_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        DE_1_SKIP_POST = RES_BLOCK(DE_1_SKIP, Block_Num=1, scope="RES_BLOCK5")

        _,_,_,Cha3 = EN_0.get_shape()
        DE_1_UP = deconv(DE_1_SKIP_POST, Cha3, kernel=3, stride=2, use_bias=True, scope='deconv_3')
        #DE_1_UP = instance_norm(DE_1_UP, scope='ins3')
        DE_1_UP = lrelu(DE_1_UP, alpha=0.1)
        #print(DE_1.shape)
        #print(DE_1_UP.shape)
        EN_0_POST = RES_BLOCK(EN_0, Block_Num=1, scope="RES_BLOCK6")
        DE_0_SKIP = DE_1_UP + EN_0_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        DE_0_SKIP_POST = RES_BLOCK(DE_0_SKIP, Block_Num=1, scope="RES_BLOCK7")

        return DE_3, DE_2_SKIP_POST, DE_1_SKIP_POST, DE_0_SKIP_POST

def ATT_BLOCK(FM_FOR_REF, MSK, MSK_FM, scope="ATB"):
    
    with tf.variable_scope(scope):

        _,_,_,Cha = FM_FOR_REF.get_shape()

        MSK_FM_POST = RES_BLOCK(MSK_FM, Block_Num=1, scope="RES_BLOCK1") #90*120*256
        #print(DE_3.shape)
        in_fm = tf.concat([MSK,MSK_FM_POST], axis=3)

        Gamma = conv(in_fm, Cha, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv1")
        #Gamma = instance_norm(Gamma, scope='ins1')
        Gamma = sigmoid(Gamma)

        Beta = conv(in_fm, Cha, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv2")
        #Beta = instance_norm(Beta, scope='ins2')
        Beta = lrelu(Beta, alpha=0.1)

        FM_FOR_REF_POST = RES_BLOCK(FM_FOR_REF, Block_Num=1, scope="RES_BLOCK2") #90*120*256

        ATT_FM_MUL = FM_FOR_REF_POST * Gamma
        ATT_FM_MUL_POST = RES_BLOCK(ATT_FM_MUL, Block_Num=1, scope="RES_BLOCK3") #90*120*256
        ATT_FM_SUM = ATT_FM_MUL_POST + Beta
        ATT_FM_SUM_POST = RES_BLOCK(ATT_FM_SUM, Block_Num=1, scope="RES_BLOCK4") #90*120*256

        return ATT_FM_SUM_POST, MSK_FM_POST

def RIGHT_REF_BLOCK(DE_3, DE_2_SKIP_POST, DE_1_SKIP_POST, DE_0_SKIP_POST, scope="RRB"):
    
    with tf.variable_scope(scope):
        EN_0 = RES_BLOCK(DE_0_SKIP_POST, Block_Num=3, scope="RES_BLOCK1") #90*120*256
        #print(DE_3.shape)

        _,_,_,Cha1 = DE_1_SKIP_POST.get_shape()
        EN_1_DOWN = conv(EN_0, Cha1, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv1") #180*240*128
        #EN_1_DOWN = instance_norm(EN_1_DOWN, scope='ins1')
        EN_1_DOWN = lrelu(EN_1_DOWN, alpha=0.1)
        #print(DE_1.shape)
        DE_1_POST = RES_BLOCK(DE_1_SKIP_POST, Block_Num=1, scope="RES_BLOCK2")
        EN_1_SKIP = EN_1_DOWN + DE_1_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        EN_1_SKIP_POST = RES_BLOCK(EN_1_SKIP, Block_Num=1, scope="RES_BLOCK3")

        _,_,_,Cha2 = DE_2_SKIP_POST.get_shape()
        EN_2_DOWN = conv(EN_1_SKIP_POST, Cha2, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv2") 
        #EN_2_DOWN = instance_norm(EN_2_DOWN, scope='ins2')
        EN_2_DOWN = lrelu(EN_2_DOWN, alpha=0.1)
        #print(DE_1.shape)
        DE_2_POST = RES_BLOCK(DE_2_SKIP_POST, Block_Num=1, scope="RES_BLOCK4")
        EN_2_SKIP = EN_2_DOWN + DE_2_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        EN_2_SKIP_POST = RES_BLOCK(EN_2_SKIP, Block_Num=1, scope="RES_BLOCK5")

        _,_,_,Cha3 = DE_3.get_shape()
        EN_3_DOWN = conv(EN_2_SKIP_POST, Cha3, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv3") 
        #EN_3_DOWN = instance_norm(EN_3_DOWN, scope='ins3')
        EN_3_DOWN = lrelu(EN_3_DOWN, alpha=0.1)
        #print(DE_1.shape)
        DE_3_POST = RES_BLOCK(DE_3, Block_Num=1, scope="RES_BLOCK6")
        EN_3_SKIP = EN_3_DOWN + DE_3_POST #90*120*(256+256)
        #print(DE_2_SKIP.shape)
        EN_3_SKIP_POST = RES_BLOCK(EN_3_SKIP, Block_Num=1, scope="RES_BLOCK7")

        return EN_0, EN_1_SKIP_POST, EN_2_SKIP_POST, EN_3_SKIP_POST

def DIM_Extend_Module(x, scope="DEM"):
    #print(x.shape)
    x = tf.expand_dims(x, -1)
    #print(x.shape)
    output = tf.concat([x, x, x], axis=3)
    #print(output.shape)
    return output

def Pooling_Conca_Module(x, scope="PCM"):
    with tf.variable_scope(scope):
        _,_,_,Cha = x.get_shape()
        Route_A = conv(x, Cha, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="Route_A")
        #Route_A = instance_norm(Route_A, scope='ins1')
        Route_A = lrelu(Route_A, alpha=0.1)
        Route_A = x * Route_A
        Route_B = conv(x, Cha, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="Route_B")
        #Route_B = instance_norm(Route_B, scope='ins2')
        Route_B = lrelu(Route_B, alpha=0.1)
        Route_B = x * Route_B
        Route_C = conv(x, Cha, kernel=5, stride=1, pad=2, pad_type='reflect', use_bias=True, scope="Route_C")
        #Route_C = instance_norm(Route_C, scope='ins3')
        Route_C = lrelu(Route_C, alpha=0.1)
        Route_C = x * Route_C
        Route_D = conv(x, Cha, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=True, scope="Route_D")
        #Route_D = instance_norm(Route_D, scope='ins4')
        Route_D = lrelu(Route_D, alpha=0.1)
        Route_D = x * Route_D
        CO_CON = tf.concat([x, Route_A, Route_B, Route_C, Route_D], 3)
        output = conv(CO_CON, Cha, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv1")
        #output = instance_norm(output, scope='ins5')
        output = lrelu(output, alpha=0.1)
        return output

def Dila_Layer(x, channels, scope="DL"):
    with tf.variable_scope(scope):
        Route_A = dila_conv(x, channels, dr=1, use_bias=True, scope='Route_A')
        #Route_A = instance_norm(Route_A, scope='ins1')
        Route_B = dila_conv(x, channels, dr=2, use_bias=True, scope='Route_B')
        #Route_B = instance_norm(Route_B, scope='ins2')
        Route_C = dila_conv(x, channels, dr=3, use_bias=True, scope='Route_C')
        #Route_C = instance_norm(Route_C, scope='ins3')
        CO_CON = tf.concat([Route_A, Route_B, Route_C], 3)
        CO_CON = relu(CO_CON, alpha=0.1)
        output = conv(CO_CON, channels, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv1")
        #output = instance_norm(output, scope='ins4')
        output = lrelu(output, alpha=0.1)
        return output

def blockLayer(x, channels, scope="BL"):
    with tf.variable_scope(scope):
        output = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="block_conv")
        #output = instance_norm(output, scope='ins1')
        return lrelu(output, alpha=0.1)

def Local_RDBA(x, L_CHs=64, L_layers=8, scope="LRDBA"):
    with tf.variable_scope(scope):
        outputs = [x]
        #rates = [1]*L_layers
        for i in range(L_layers):
            output_in = Dila_Layer(tf.concat(outputs[:i],3) if i>=1 else x, L_CHs, scope=scope+str(i+1))
            outputs.append(output_in)

        output = tf.concat(outputs, 3)
        output = conv(output, L_CHs, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv")
        #output = instance_norm(output, scope='ins1')
        output = lrelu(output, alpha=0.1)
        return x + output

def Global_RDBA(x, G_CHs=64, G_layers=4, L_Layers=8, scope="GRDBA"):
    with tf.variable_scope(scope):
        outputs = [x]
        for i in range(G_layers):
            output_in = Local_RDBA(tf.concat(outputs[:i],3) if i>=1 else x, G_CHs, L_Layers, scope=scope+str(i+1))
            outputs.append(output_in)

        output = tf.concat(outputs, 3)
        output = conv(output, G_CHs, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv")
        #output = instance_norm(output, scope='ins1')
        output = lrelu(output, alpha=0.1)
        return output

def Local_RDBB(x, L_CHs=64, L_layers=8, scope="LRDBB"):
    with tf.variable_scope(scope):
        outputs = [x]
        for i in range(L_layers):
            output_in = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, L_CHs, scope=scope+str(i+1))
            outputs.append(output_in)

        output = tf.concat(outputs, 3)
        output = conv(output, L_CHs, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv")
        #output = instance_norm(output, scope='ins1')
        output = lrelu(output, alpha=0.1)
        return x + output

def Global_RDBB(x, G_CHs=64, G_layers=4, L_Layers=8, scope="GRDBB"):
    with tf.variable_scope(scope):
        outputs = [x]
        for i in range(G_layers):
            output_in = Local_RDBB(tf.concat(outputs[:i],3) if i>=1 else x, G_CHs, L_Layers, scope=scope+str(i+1))
            outputs.append(output_in)

        output = tf.concat(outputs, 3)
        output = conv(output, G_CHs, kernel=1, stride=1, pad=0, pad_type='reflect', use_bias=True, scope="conv")
        #output = instance_norm(output, scope='ins1')
        output = lrelu(output, alpha=0.1)
        return output

##################################################################################
# Reconstruction Loss Function
##################################################################################
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.nn.l2_loss(x - y))

    return loss

def MSE_loss(x, y):
    loss = tf.losses.mean_squared_error(x, y)

    return loss

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(tf.nn.sigmoid(real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid(fake))

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))*0.5
        fake_loss = tf.reduce_mean(tf.square(fake))*0.5

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(tf.nn.sigmoid(fake))

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(tf.nn.sigmoid(fake))

    loss = fake_loss

    return loss
