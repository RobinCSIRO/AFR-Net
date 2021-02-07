import time
import random
import collections
import vgg19
import matplotlib.pyplot as plt
from ops import *
from utils import *
from metric import *
from ssim_loss import *

#from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

class AFR_DR(object):

    def __init__(self, sess, args):
        self.model_name = "RD_AFR"  # name for checkpoint
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.fmvis_dir = args.fmvis_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.epoch_metric_dir = args.epoch_metric_dir

        self.epochs = args.epoch
        self.iteration = args.iteration
        self.gbatch_size = args.gbatch_size
        self.lbatch_size = args.lbatch_size
        self.patch_size = args.patch_size
        self.patch_stride = args.patch_stride
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.Train_W = args.Train_W
        self.Train_H = args.Train_H
        self.c_dim = 3
        self.img_shape = [self.Train_W, self.Train_H, self.c_dim]

        """ Basis of Filter Numbers """
        self.ngf = args.ngf   #32  
        self.ndf= args.ndf    #64  
        self.dropout_rate= args.dropout_rate  

        self.rec_alpha = args.alpha1
        self.per_alpha = args.alpha2  
        self.adv_alpha = args.alpha3      

        self.test_num = args.test_num

        # train
        self.init_lr = args.init_lr
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.gan_type = args.gan_type

        self.net_path = args.vgg_model_path
        self.per_net= vgg19.VGG19(self.net_path)
        CONTENT_LAYERS = {}
        for layer, weight in zip(args.content_layers,args.content_layer_weights):
            CONTENT_LAYERS[layer] = weight
        self.CONTENT_LAYERS = collections.OrderedDict(sorted(CONTENT_LAYERS.items()))

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.fmvis_dir = os.path.join(self.fmvis_dir, self.model_dir)
        check_folder(self.fmvis_dir)

        print()
        print("============================================")
        print("# gan type : ", self.gan_type)
        print("# Image_Width : ", self.Train_W)
        print("# Image_Height : ", self.Train_H)
        print("# gbatch_size : ", self.gbatch_size)
        print("# lbatch_size : ", self.lbatch_size)
        print("# patch_size : ", self.patch_size)
        print("# patch_stride : ", self.patch_stride)
        print("# epoch : ", self.epochs)
        print("# iteration per epoch : ", self.iteration)
        print("============================================")

    ##################################################################################
    # Generator
    #################################################################################
    #############Part-I Shared_Encoder########################
    def SHARE_ENC(self, input_img, reuse=False, scope="SEN"): 
        with tf.variable_scope(scope, reuse=reuse):
            """ STEM_Block """
            x_in = conv(input_img, self.ngf, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=True, scope="conv1")
            #x_in = instance_norm(x_in, scope='ins1')
            x_in = lrelu(x_in, alpha=0.1)
            #print(x.shape)
            """ STEM_Block """

            """ Four_Base_Blocks for Refinement """
            EN_0 = conv(x_in, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv2")
            #EN_0 = instance_norm(EN_0, scope='ins2')
            EN_0 = lrelu(EN_0, alpha=0.1)
            #print(EN_1_DOWN.shape)
            EN_0 = Global_RDBB(EN_0, G_CHs=self.ngf, G_layers=2, L_Layers=3, scope="EN_0") #360*480*64
            #print(EN_1.shape)

            EN_1_DOWN = conv(EN_0, self.ngf*2, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv3")
            #EN_1_DOWN = instance_norm(EN_1_DOWN, scope='ins3')
            EN_1_DOWN = lrelu(EN_1_DOWN, alpha=0.1)
            #print(EN_1_DOWN.shape)
            EN_1 = Global_RDBB(EN_1_DOWN, G_CHs=self.ngf*2, G_layers=2, L_Layers=3, scope="EN_1") #360*480*64
            #print(EN_1.shape)

            EN_2_DOWN = conv(EN_1, self.ngf*2, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv4")
            #EN_2_DOWN = instance_norm(EN_2_DOWN, scope='ins4')
            EN_2_DOWN = lrelu(EN_2_DOWN, alpha=0.1)
            #print(EN_2_DOWN.shape)
            EN_2 = Global_RDBB(EN_2_DOWN, G_CHs=self.ngf*2, G_layers=2, L_Layers=3, scope="EN_2") #360*480*64
            #print(EN_2.shape)

            EN_3_DOWN = conv(EN_2, self.ngf*4, kernel=3, stride=2, pad=1, pad_type='reflect', use_bias=True, scope="conv5")
            #EN_3_DOWN = instance_norm(EN_3_DOWN, scope='ins5')
            EN_3_DOWN = lrelu(EN_3_DOWN, alpha=0.1)
            #print(EN_3_DOWN.shape)
            EN_3 = Global_RDBB(EN_3_DOWN, G_CHs=self.ngf*4, G_layers=2, L_Layers=3, scope="EN_3") #360*480*64
            #print(EN_3.shape)
            """ Four_Base_Blocks for Refinement """

            return x_in, EN_0, EN_1, EN_2, EN_3
    #############Part-I Shared_Encoder########################

    #############Part-I Mask_Decoder#######################
    def MSK_DEC(self, IN_FM, EN_0, EN_1, EN_2, EN_3, reuse=False, scope="MDE"):  
        with tf.variable_scope(scope, reuse=reuse):

            DE_3 = RES_BLOCK(EN_3, Block_Num=3, scope="RES_BLOCK1") #90*120*256
            #print(DE_3.shape)

            DE_3_UP = deconv(DE_3, self.ngf*2, kernel=3, stride=2, use_bias=True, scope='deconv_1')
            #DE_3_UP = instance_norm(DE_3_UP, scope='ins1')
            DE_3_UP = lrelu(DE_3_UP, alpha=0.1)
            DE_3_UP = Global_RDBB(DE_3_UP, G_CHs=self.ngf*2, G_layers=2, L_Layers=2, scope="DE_3") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_2_POST = RES_BLOCK(EN_2, Block_Num=1, scope="RES_BLOCK2")
            DE_2_SKIP = DE_3_UP + EN_2_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_2_SKIP_POST = RES_BLOCK(DE_2_SKIP, Block_Num=1, scope="RES_BLOCK3")

            DE_2_UP = deconv(DE_2_SKIP_POST, self.ngf*2, kernel=3, stride=2, use_bias=True, scope='deconv_2')
            #DE_2_UP = instance_norm(DE_2_UP, scope='ins2')
            DE_2_UP = lrelu(DE_2_UP, alpha=0.1)
            DE_2_UP = Global_RDBB(DE_2_UP, G_CHs=self.ngf*2, G_layers=2, L_Layers=2, scope="DE_2") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_1_POST = RES_BLOCK(EN_1, Block_Num=1, scope="RES_BLOCK4")
            DE_1_SKIP = DE_2_UP + EN_1_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_1_SKIP_POST = RES_BLOCK(DE_1_SKIP, Block_Num=1, scope="RES_BLOCK5")

            DE_1_UP = deconv(DE_1_SKIP_POST, self.ngf, kernel=3, stride=2, use_bias=True, scope='deconv_3')
            #DE_1_UP = instance_norm(DE_1_UP, scope='ins3')
            DE_1_UP = lrelu(DE_1_UP, alpha=0.1)
            DE_1_UP = Global_RDBB(DE_1_UP, G_CHs=self.ngf, G_layers=2, L_Layers=2, scope="DE_1") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_0_POST = RES_BLOCK(EN_0, Block_Num=1, scope="RES_BLOCK6")
            DE_0_SKIP = DE_1_UP + EN_0_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_0_SKIP_POST = RES_BLOCK(DE_0_SKIP, Block_Num=1, scope="RES_BLOCK7")

            #print(DE_3.shape)
            Pre_Out = conv(DE_0_SKIP_POST, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv1")
            #Pre_Out = instance_norm(Pre_Out, scope='ins4')
            Pre_Out = lrelu(Pre_Out, alpha=0.1)
            Pre_Out = IN_FM + Pre_Out
            ##########Full_Size_Output###########
            Full_Out = conv(DE_0_SKIP_POST, 1, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, scope="conv2")
            #Full_Out = FU_In + Full_Out
            #print(Full_Out.shape)
            #print("============================================")
            return Full_Out, Pre_Out
    #############Part-I Mask_Decoder#######################

    #############Part-I FM_Refinement#######################
    def FM_REF(self, MSK, MSK_FM, EN_0, EN_1, EN_2, EN_3, reuse=False, scope="FMREF"):  
        with tf.variable_scope(scope, reuse=reuse):

            """ First Refinement """
            REF1_DE_0, REF1_DE_1, REF1_DE_2, REF1_DE_3 = LEFT_REF_BLOCK(EN_0, EN_1, EN_2, EN_3, scope="LRB1") #90*120*256
            ATT_FM1, MSK_FM_POST1 = ATT_BLOCK(REF1_DE_3, MSK, MSK_FM, scope="ATB1") #90*120*256
            REF1_EN_0, REF1_EN_1, REF1_EN_2, REF1_EN_3 = RIGHT_REF_BLOCK(REF1_DE_0, REF1_DE_1, REF1_DE_2, ATT_FM1, scope="RRB1") 
            #print(DE_3.shape)
            """ First Refinement """

            """ Second Refinement """
            REF2_DE_0, REF2_DE_1, REF2_DE_2, REF2_DE_3 = LEFT_REF_BLOCK(REF1_EN_0, REF1_EN_1, REF1_EN_2, REF1_EN_3, scope="LRB2") 
            ATT_FM2, MSK_FM_POST2 = ATT_BLOCK(REF2_DE_3, MSK, MSK_FM_POST1, scope="ATB2") #90*120*256
            REF2_EN_0, REF2_EN_1, REF2_EN_2, REF2_EN_3 = RIGHT_REF_BLOCK(REF2_DE_0, REF2_DE_1, REF2_DE_2, ATT_FM2, scope="RRB2") 
            #print(DE_3.shape)
            """ Second Refinement """

            """ Third Refinement """
            REF3_DE_0, REF3_DE_1, REF3_DE_2, REF3_DE_3 = LEFT_REF_BLOCK(REF2_EN_0, REF2_EN_1, REF2_EN_2, REF2_EN_3, scope="LRB3") 
            ATT_FM3, _ = ATT_BLOCK(REF3_DE_3, MSK, MSK_FM_POST2, scope="ATB3") #90*120*256
            REF3_EN_0, REF3_EN_1, REF3_EN_2, REF3_EN_3 = RIGHT_REF_BLOCK(REF3_DE_0, REF3_DE_1, REF3_DE_2, ATT_FM3, scope="RRB3") 
            #print(DE_3.shape)
            """ Third Refinement """

            """ Feature Map Visualization """
            #print(REF1_DE_3.shape)
            FM_B_AM_GAP = tf.reduce_mean(REF3_DE_3, axis=-1, keepdims=False)
            #print(FM_B_AM_GAP.shape)
            FM_A_AM_GAP = tf.reduce_mean(ATT_FM3, axis=-1, keepdims=False)
            FM_B_AM_GMP = tf.reduce_max(REF3_DE_3, axis=-1, keepdims=False)
            FM_A_AM_GMP = tf.reduce_max(ATT_FM3, axis=-1, keepdims=False)
            #print(DE_3.shape)
            """ Feature Map Visualization """

            return REF3_EN_0, REF3_EN_1, REF3_EN_2, REF3_EN_3, FM_B_AM_GAP, FM_A_AM_GAP, FM_B_AM_GMP, FM_A_AM_GMP
    #############Part-I FM_Refinement#######################


    #############Part-I Derain with Refined FM#######################
    def DRA_DEC(self, IN_FM, REF3_EN_0, REF3_EN_1, REF3_EN_2, REF3_EN_3, reuse=False, scope="CDE"):  
        with tf.variable_scope(scope, reuse=reuse):

            DE_3 = RES_BLOCK(REF3_EN_3, Block_Num=3, scope="RES_BLOCK1") #90*120*256
            #print(DE_3.shape)

            DE_3_UP = deconv(DE_3, self.ngf*2, kernel=3, stride=2, use_bias=True, scope='deconv_1')
            #DE_3_UP = instance_norm(DE_3_UP, scope='ins1')
            DE_3_UP = lrelu(DE_3_UP, alpha=0.1)
            DE_3_UP = Global_RDBB(DE_3_UP, G_CHs=self.ngf*2, G_layers=2, L_Layers=3, scope="DE_3") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_2_POST = RES_BLOCK(REF3_EN_2, Block_Num=3, scope="RES_BLOCK2")
            DE_2_SKIP = DE_3_UP + EN_2_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_2_SKIP_POST = RES_BLOCK(DE_2_SKIP, Block_Num=3, scope="RES_BLOCK3")

            DE_2_UP = deconv(DE_2_SKIP_POST, self.ngf*2, kernel=3, stride=2, use_bias=True, scope='deconv_2')
            #DE_2_UP = instance_norm(DE_2_UP, scope='ins2')
            DE_2_UP = lrelu(DE_2_UP, alpha=0.1)
            DE_2_UP = Global_RDBB(DE_2_UP, G_CHs=self.ngf*2, G_layers=2, L_Layers=3, scope="DE_2") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_1_POST = RES_BLOCK(REF3_EN_1, Block_Num=3, scope="RES_BLOCK4")
            DE_1_SKIP = DE_2_UP + EN_1_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_1_SKIP_POST = RES_BLOCK(DE_1_SKIP, Block_Num=3, scope="RES_BLOCK5")

            DE_1_UP = deconv(DE_1_SKIP_POST, self.ngf, kernel=3, stride=2, use_bias=True, scope='deconv_3')
            #DE_1_UP = instance_norm(DE_1_UP, scope='ins3')
            DE_1_UP = lrelu(DE_1_UP, alpha=0.1)
            DE_1_UP = Global_RDBB(DE_1_UP, G_CHs=self.ngf, G_layers=2, L_Layers=3, scope="DE_1") #90*120*256
            #print(DE_1.shape)
            #print(DE_1_UP.shape)
            EN_0_POST = RES_BLOCK(REF3_EN_0, Block_Num=3, scope="RES_BLOCK6")
            DE_0_SKIP = DE_1_UP + EN_0_POST #90*120*(256+256)
            #print(DE_2_SKIP.shape)
            DE_0_SKIP_POST = RES_BLOCK(DE_0_SKIP, Block_Num=3, scope="RES_BLOCK7")

            #print(DE_3.shape)
            Pre_Out = conv(DE_0_SKIP_POST, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv1")
            #Pre_Out = instance_norm(Pre_Out, scope='ins4')
            Pre_Out = lrelu(Pre_Out, alpha=0.1)
            Pre_Out = IN_FM + Pre_Out

            ##########Full_Size_Output###########
            Full_Out = conv(Pre_Out, self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, scope="conv2")
            #Full_Out = FU_In + Full_Out
            #print(Full_Out.shape)
            ##########1/2_Size_Output###########
            HA_Out = conv(DE_1_SKIP_POST, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv3")
            HA_Out = lrelu(HA_Out, alpha=0.1)
            HA_Out = conv(HA_Out, self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, scope="conv4")
            #HA_Out = HA_In + HA_Out
            #print(HA_Out.shape)
            ##########1/4_Size_Output###########
            HAHA_Out = conv(DE_2_SKIP_POST, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv5")
            HAHA_Out = lrelu(HAHA_Out, alpha=0.1)
            HAHA_Out = conv(HAHA_Out, self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, scope="conv6")
            #HAHA_Out = HAHA_In + HAHA_Out
            #print(HAHA_Out.shape)
            ##########1/8_Size_Output###########
            HAHAHA_Out = conv(DE_3, self.ngf, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=True, scope="conv7")
            HAHAHA_Out = lrelu(HAHAHA_Out, alpha=0.1)
            HAHAHA_Out = conv(HAHAHA_Out, self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, scope="conv8")
            #HAHA_Out = HAHA_In + HAHA_Out
            #print(HAHA_Out.shape)
            #print("============================================")
            return Full_Out, HA_Out, HAHA_Out, HAHAHA_Out
    #############Part-I Derain with Refined FM#######################

    #############Part-I Feature Map Visualization#######################
    def FM_VIS(self, EN_0, EN_1, EN_2, EN_3, REF3_EN_0, REF3_EN_1, REF3_EN_2, REF3_EN_3):  
        """ Visualization of FM before Refinement """
        #GAP_FM
        EN0_B_REF_GAP = tf.reduce_mean(EN_0, axis=-1, keepdims=False)
        EN1_B_REF_GAP = tf.reduce_mean(EN_1, axis=-1, keepdims=False)
        EN2_B_REF_GAP = tf.reduce_mean(EN_2, axis=-1, keepdims=False)
        EN3_B_REF_GAP = tf.reduce_mean(EN_3, axis=-1, keepdims=False)
        #GMP_FM
        EN0_B_REF_GMP = tf.reduce_max(EN_0, axis=-1, keepdims=False)
        EN1_B_REF_GMP = tf.reduce_max(EN_1, axis=-1, keepdims=False)
        EN2_B_REF_GMP = tf.reduce_max(EN_2, axis=-1, keepdims=False)
        EN3_B_REF_GMP = tf.reduce_max(EN_3, axis=-1, keepdims=False)
        """ Visualization of FM before Refinement """

        """ Visualization of FM after Refinement """
        #GAP_FM
        EN0_A_REF_GAP = tf.reduce_mean(REF3_EN_0, axis=-1, keepdims=False)
        EN1_A_REF_GAP = tf.reduce_mean(REF3_EN_1, axis=-1, keepdims=False)
        EN2_A_REF_GAP = tf.reduce_mean(REF3_EN_2, axis=-1, keepdims=False)
        EN3_A_REF_GAP = tf.reduce_mean(REF3_EN_3, axis=-1, keepdims=False)
        #GMP_FM
        EN0_A_REF_GMP = tf.reduce_max(REF3_EN_0, axis=-1, keepdims=False)
        EN1_A_REF_GMP = tf.reduce_max(REF3_EN_1, axis=-1, keepdims=False)
        EN2_A_REF_GMP = tf.reduce_max(REF3_EN_2, axis=-1, keepdims=False)
        EN3_A_REF_GMP = tf.reduce_max(REF3_EN_3, axis=-1, keepdims=False)
        """ Visualization of FM before Refinement """
        return EN0_B_REF_GAP, EN1_B_REF_GAP, EN2_B_REF_GAP, EN3_B_REF_GAP, EN0_B_REF_GMP, EN1_B_REF_GMP, EN2_B_REF_GMP, EN3_B_REF_GMP, EN0_A_REF_GAP, EN1_A_REF_GAP, EN2_A_REF_GAP, EN3_A_REF_GAP, EN0_A_REF_GMP, EN1_A_REF_GMP, EN2_A_REF_GMP, EN3_A_REF_GMP
    #############Part-I Feature Map Visualization#######################


    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        self.rainy = tf.placeholder(tf.float32, [self.gbatch_size, self.Train_H, self.Train_W, self.c_dim], name='rainy_images')
        self.GT = tf.placeholder(tf.float32, [self.gbatch_size, self.Train_H, self.Train_W, self.c_dim], name='ground_truth')
        self.mask = tf.placeholder(tf.float32, [self.gbatch_size, self.Train_H, self.Train_W, 1], name='clean_masks')
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
        self.vis1 = self.rainy
        self.vis2 = self.GT

        ################Input image resize for multi-ouput#########################
        self.FU_RA = self.rainy
        self.HA_RA = tf.image.resize_bilinear(images=self.rainy, size=(int(self.Train_H / 2),int(self.Train_W / 2)))
        self.HAHA_RA = tf.image.resize_bilinear(images=self.rainy, size=(int(self.Train_H / 4),int(self.Train_W / 4)))
        self.HAHAHA_RA = tf.image.resize_bilinear(images=self.rainy, size=(int(self.Train_H / 8),int(self.Train_W / 8)))
        self.FU_GT = self.GT
        self.HA_GT = tf.image.resize_bilinear(images=self.GT, size=(int(self.Train_H / 2),int(self.Train_W / 2)))
        self.HAHA_GT = tf.image.resize_bilinear(images=self.GT, size=(int(self.Train_H / 4),int(self.Train_W / 4)))
        self.HAHAHA_GT = tf.image.resize_bilinear(images=self.GT, size=(int(self.Train_H / 8),int(self.Train_W / 8)))
        ################Input image resize for multi-ouput#########################

        """ Multi_Output with Generator Formulation """
        ################Running Shared_Encoder################
        self.SKIP_FM, self.SEN0, self.SEN1, self.SEN2, self.SEN3 = self.SHARE_ENC(self.rainy)
        ################Running Shared_Encoder################

        ################Running Mask_Decoder################
        self.MASK, self.INIT_MFM = self.MSK_DEC(self.SKIP_FM, self.SEN0, self.SEN1, self.SEN2, self.SEN3)
        ################Running Mask_Decoder################

        ################Running FM_Refinement################
        self.REN0, self.REN1, self.REN2, self.REN3, _, _, _, _ = self.FM_REF(self.MASK, self.INIT_MFM, self.SEN0, self.SEN1, self.SEN2, self.SEN3)
        ################Running FM_Refinement################

        ################Running Derainer################
        self.FU_DR, self.HA_DR, self.HAHA_DR, self.HAHAHA_DR = self.DRA_DEC(self.SKIP_FM, self.REN0, self.REN1, self.REN2, self.REN3)
        ################Running Derainer################


        self.vis3 = 1-self.mask
        self.vis5 = self.FU_DR

        """ Multiple Loss Formulation """
        ################Perceptual loss part################
        self.GT_pre = self.per_net.preprocess(self.GT)
        self.clean_pre = self.per_net.preprocess(self.FU_DR)
        layers_output = self.per_net.feed_forward(self.GT_pre, scope='GT')
        #print('The following is the shape of content layer:')
        #print(content_layers.shape)
        self.GT_FM = {}
        for id in self.CONTENT_LAYERS:
            self.GT_FM[id] = layers_output[id]
        self.Clean_FM = self.per_net.feed_forward(self.clean_pre, scope='Processed')
        L_content = 0
        for id in self.Clean_FM:
            if id in self.CONTENT_LAYERS:
                ## content loss ##

                F = self.Clean_FM[id]             # content feature of x
                P = self.GT_FM[id]             # content feature of p

                b, h, w, d = F.get_shape()  # first return value is batch size (must be one)
                b = b.value                 # batch size
                N = h.value*w.value         # product of width and height
                M = d.value                 # number of filters

                w = self.CONTENT_LAYERS[id] # weight for this layer

                L_content += w * 2 * tf.nn.l2_loss(F-P) / (b*N*M)
        self.PER_Loss = L_content*0.005

        ################Mask_Branch loss part################
        self.MSK_Loss = L2_loss(self.MASK, self.mask)*0.1
        ################Derains_Branch loss part################
        self.FU_loss = L2_loss(self.FU_DR, self.FU_GT)
        self.HA_loss = L2_loss(self.HA_DR, self.HA_GT)*0.8
        self.HAHA_loss = L2_loss(self.HAHA_DR, self.HAHA_GT)*0.6
        self.HAHAHA_loss = L2_loss(self.HAHAHA_DR, self.HAHAHA_GT)*0.4
        self.DR_Loss = self.FU_loss + self.HA_loss + self.HAHA_loss + self.HAHAHA_loss
        ################Overall loss part################
        self.derain_loss = self.DR_Loss + self.MSK_Loss + self.PER_Loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()

        g_vars_sen = [var for var in t_vars if 'SEN' in var.name]
        g_vars_mde = [var for var in t_vars if 'MDE' in var.name]
        g_vars_fmref = [var for var in t_vars if 'FMREF' in var.name]
        g_vars_cde = [var for var in t_vars if 'CDE' in var.name]
        g_vars = [g_vars_sen, g_vars_mde, g_vars_fmref, g_vars_cde]

        # optimizers
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2).minimize(self.derain_loss, var_list=g_vars, global_step=self.global_step)

        """" Testing_For_Test_Mode """
        # for test
        self.TestR = tf.placeholder(tf.float32, [1, self.Train_H, self.Train_W, self.c_dim], name='TestR')
        self.TestG = tf.placeholder(tf.float32, [1, self.Train_H, self.Train_W, self.c_dim], name='TestG')
        self.Tvis1 = self.TestR
        self.Tvis2 = self.TestG

        ################Running Shared_Encoder################
        self.TSKIP_FM, self.TSEN0, self.TSEN1, self.TSEN2, self.TSEN3 = self.SHARE_ENC(self.TestR, reuse=True)
        ################Running Shared_Encoder################

        ################Running Mask_Decoder################
        self.TMASK, self.TINIT_MFM = self.MSK_DEC(self.TSKIP_FM, self.TSEN0, self.TSEN1, self.TSEN2, self.TSEN3, reuse=True)
        ################Running Mask_Decoder################

        ################Running FM_Refinement################
        self.TREN0, self.TREN1, self.TREN2, self.TREN3, self.FMBGAP, self.FMAGAP, self.FMBGMP, self.FMAGMP = self.FM_REF(self.TMASK, self.TINIT_MFM, self.TSEN0, self.TSEN1, self.TSEN2, self.TSEN3, reuse=True)
        ################Running FM_Refinement################

        ################Running Refined_FM Visualization################
        self.EN0_B_REF_GAP, self.EN1_B_REF_GAP, self.EN2_B_REF_GAP, self.EN3_B_REF_GAP, self.EN0_B_REF_GMP, self.EN1_B_REF_GMP, self.EN2_B_REF_GMP, self.EN3_B_REF_GMP,   self.EN0_A_REF_GAP, self.EN1_A_REF_GAP, self.EN2_A_REF_GAP, self.EN3_A_REF_GAP, self.EN0_A_REF_GMP, self.EN1_A_REF_GMP, self.EN2_A_REF_GMP, self.EN3_A_REF_GMP = self.FM_VIS(self.TSEN0, self.TSEN1, self.TSEN2, self.TSEN3, self.TREN0, self.TREN1, self.TREN2, self.TREN3)
        ################Running Refined_FM Visualization################

        ################Running Derainer################
        self.TFU_DR, _, _, _  = self.DRA_DEC(self.TSKIP_FM, self.TREN0, self.TREN1, self.TREN2, self.TREN3, reuse=True)
        ################Running Derainer################

        self.Tvis5 = self.TFU_DR

        """ Summary """
        # self.l_sum = tf.summary.scalar("l_loss", self.Re_loss)
        # self.p_sum = tf.summary.scalar("per_loss", self.Perceptual_loss)
        self.loss_sum = tf.summary.scalar("Loss", self.derain_loss)


    ##################################################################################
    # Train
    ##################################################################################

    def train(self):

        train_rainy_data = glob(os.path.join('../../Derain_Data/train/data/','*.png'))
        num_examples = len(train_rainy_data)
        print("============================")
        print(num_examples)
        print("============================")
        train_rainy_path = '../../Derain_Data/train/data/'
        train_gt_path = '../../Derain_Data/train/gt/'

        test_rainy_path = '../../Derain_Data/test_a/data/'
        test_gt_path = '../../Derain_Data/test_a/gt/'


        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, counter = self.load(self.checkpoint_dir)

        if could_load:
            #print(checkpoint_counter)
            epoch = (counter * self.gbatch_size) // num_examples
            iterations = counter - epoch*(num_examples // self.gbatch_size)
            print(" [*] Load SUCCESS")
        else:
            epoch = 0
            iterations = 0
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        curr_lr = self.init_lr
        while epoch < self.epochs:
            if self.decay_flag:
                curr_lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epochs-epoch) / (self.epochs-self.decay_epoch)

            while (iterations * self.gbatch_size) < num_examples:

                data_id = iterations * self.gbatch_size
                #print('==============================')
                batch_rimages, batch_gimages, _, batch_masks =read_tridata(train_rainy_path, train_gt_path, self.Train_W, self.Train_H, data_id, self.gbatch_size)
                train_feed_dict = {self.rainy: batch_rimages, self.GT: batch_gimages, self.mask: batch_masks,  self.lr: curr_lr}

                # update Derain network
                _, summary_str, Total_loss, DR_Loss, MSK_Loss, PER_Loss, counter = self.sess.run([self.g_optim, self.loss_sum, self.derain_loss, self.DR_Loss, self.MSK_Loss, self.PER_Loss, self.global_step], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)


                print("Epoch: [%4d] [%2d/%2d] time: %4.4f, Total_Loss: %.5f, DR_Loss: %.5f, MSK_Loss: %.5f, PER_Loss: %.5f" \
                      % (epoch+1, iterations+1, self.iteration, time.time() - start_time, Total_loss, DR_Loss, MSK_Loss, PER_Loss))

                line1 = "Epoch:[%3d], Counter:[%6d], Total_Loss: %.5f, DR_Loss: %.5f, MSK_Loss: %.5f, PER_Loss: %.5f \n" % (epoch+1, counter, Total_loss, DR_Loss, MSK_Loss, PER_Loss)
                with open('logs_loss.txt', 'a') as f:
                    f.write(line1)
                iterations += 1
            ############################PSNR and SSIM################################################
            if np.mod(epoch+1, 1) == 0:
                ######Fine_Clean Part##############
                FTe_psnr = []
                FTe_ssim = []
                for Fsample_ID in range(58):
                    FTe_batch_r, FTe_batch_g,_,_ = read_tridata(test_rainy_path, test_gt_path, self.Train_W, self.Train_H, Fsample_ID, 1)
                    F_DR_img, F_msk = self.sess.run([self.TFU_DR,self.TMASK], feed_dict={self.TestR: FTe_batch_r,self.TestG: FTe_batch_g})
                    FGT_imgs = FTe_batch_g[0,:,:,:]
                    FGT_imgs = ((FGT_imgs+1.0)*127.5).astype(np.uint8)
                    F_DR_img = F_DR_img[0,:,:,:]
                    F_DR_img = post_ouput(F_DR_img)
                    F_DR_img = ((F_DR_img+1.0)*127.5).astype(np.uint8)
                    F_MSK = F_msk[0,:,:,:]
                    F_MSK = att_post_ouput(F_MSK)
                    F_MSK = ((F_MSK)*255.0).astype(np.uint8)
                    Att_imgs = F_msk[0,:,:,:]
                    #print(Att_imgs.shape)
                    Att_imgs = att_post_ouput(Att_imgs)
                    if np.mod(epoch+1, 10) == 0:
                        cv2.imwrite(self.sample_dir+'/' + 'DRA_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), F_DR_img)
                    if np.mod(epoch+1, 50) == 0:
                        cv2.imwrite(self.sample_dir+'/' + 'MSK_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), F_MSK)
                        att_imgs = cv2.GaussianBlur(Att_imgs, (15, 15), 0)
                        plt.imsave(self.sample_dir+'/' + 'ATT_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), att_imgs, cmap='jet')
                    if np.mod(epoch+1, 100) == 0:
                        FMBGAP, FMAGAP, FMBGMP, FMAGMP = self.sess.run([self.FMBGAP, self.FMAGAP, self.FMBGMP, self.FMAGMP], feed_dict={self.TestR: FTe_batch_r,self.TestG: FTe_batch_g})
                        BGAP = FMBGAP[0,:,:]
                        AGAP = FMAGAP[0,:,:]
                        BGMP = FMBGMP[0,:,:]
                        AGMP = FMAGMP[0,:,:]
                        #BGAP_GAU = cv2.GaussianBlur(BGAP, (5, 5), 0)
                        plt.imsave(self.fmvis_dir+'/' + 'FMBGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), BGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'FMAGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), AGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'FMBGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), BGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'FMAGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), AGMP, cmap='jet')
                        EN0_B_REF_GAP, EN1_B_REF_GAP, EN2_B_REF_GAP, EN3_B_REF_GAP, EN0_B_REF_GMP, EN1_B_REF_GMP, EN2_B_REF_GMP, EN3_B_REF_GMP, EN0_A_REF_GAP, EN1_A_REF_GAP, EN2_A_REF_GAP, EN3_A_REF_GAP, EN0_A_REF_GMP, EN1_A_REF_GMP, EN2_A_REF_GMP, EN3_A_REF_GMP = self.sess.run([self.EN0_B_REF_GAP, self.EN1_B_REF_GAP, self.EN2_B_REF_GAP, self.EN3_B_REF_GAP, self.EN0_B_REF_GMP, self.EN1_B_REF_GMP, self.EN2_B_REF_GMP, self.EN3_B_REF_GMP,   self.EN0_A_REF_GAP, self.EN1_A_REF_GAP, self.EN2_A_REF_GAP, self.EN3_A_REF_GAP, self.EN0_A_REF_GMP, self.EN1_A_REF_GMP, self.EN2_A_REF_GMP, self.EN3_A_REF_GMP], feed_dict={self.TestR: FTe_batch_r,self.TestG: FTe_batch_g})
                        EN0BREFGAP = EN0_B_REF_GAP[0,:,:]
                        EN1BREFGAP = EN1_B_REF_GAP[0,:,:]
                        EN2BREFGAP = EN2_B_REF_GAP[0,:,:]
                        EN3BREFGAP = EN3_B_REF_GAP[0,:,:]
                        EN0BREFGMP = EN0_B_REF_GMP[0,:,:]
                        EN1BREFGMP = EN1_B_REF_GMP[0,:,:]
                        EN2BREFGMP = EN2_B_REF_GMP[0,:,:]
                        EN3BREFGMP = EN3_B_REF_GMP[0,:,:]
                        EN0AREFGAP = EN0_A_REF_GAP[0,:,:]
                        EN1AREFGAP = EN1_A_REF_GAP[0,:,:]
                        EN2AREFGAP = EN2_A_REF_GAP[0,:,:]
                        EN3AREFGAP = EN3_A_REF_GAP[0,:,:]
                        EN0AREFGMP = EN0_A_REF_GMP[0,:,:]
                        EN1AREFGMP = EN1_A_REF_GMP[0,:,:]
                        EN2AREFGMP = EN2_A_REF_GMP[0,:,:]
                        EN3AREFGMP = EN3_A_REF_GMP[0,:,:]
                        plt.imsave(self.fmvis_dir+'/' + 'EN0BREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN0BREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN1BREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN1BREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN2BREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN2BREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN3BREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN3BREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN0BREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN0BREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN1BREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN1BREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN2BREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN2BREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN3BREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN3BREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN0AREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN0AREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN1AREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN1AREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN2AREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN2AREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN3AREFGAP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN3AREFGAP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN0AREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN0AREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN1AREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN1AREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN2AREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN2AREFGMP, cmap='jet')
                        plt.imsave(self.fmvis_dir+'/' + 'EN3AREFGMP_Epoch_' + '%d_%d.png'%(epoch+1,Fsample_ID), EN3AREFGMP, cmap='jet')
                    Fcur_psnr = calc_psnr(FGT_imgs, F_DR_img)
                    Fcur_ssim = calc_ssim(FGT_imgs, F_DR_img)
                    FTe_psnr.append(Fcur_psnr)
                    FTe_ssim.append(Fcur_ssim)
                FTe_psnr_mean = np.mean(FTe_psnr)
                FTe_ssim_mean = np.mean(FTe_ssim)
                FTe_psnr_max = np.max(FTe_psnr)
                FTe_ssim_max = np.max(FTe_ssim)
                FTe_psnr_min = np.min(FTe_psnr)
                FTe_ssim_min = np.min(FTe_ssim)
                print('==============================')
                print('Sample number for PSNR/SSIM calculation is: %d'%(Fsample_ID+1))
                print('FTest_Metric_Value for Epoch[%3d] -- PSNR is %.4f (%.4f/%.4f) and SSIM is %.4f (%.4f/%.4f)'%(epoch+1, FTe_psnr_mean, FTe_psnr_max, FTe_psnr_min, FTe_ssim_mean, FTe_ssim_max, FTe_ssim_min))
                print('==============================')
                lineF = "Epoch:%3d, PSNR is %.4f (%.4f/%.4f) and SSIM is %.4f (%.4f/%.4f) \n" % (epoch+1, FTe_psnr_mean, FTe_psnr_max, FTe_psnr_min, FTe_ssim_mean, FTe_ssim_max, FTe_ssim_min)
                with open('FTest_metric.txt', 'a') as f:
                    f.write(lineF)
                line_PSNR = "%.4f \n" % (FTe_psnr_mean)
                with open('RD_AFR_PSNR.txt', 'a') as f:
                    f.write(line_PSNR)
                line_SSIM= "%.4f \n" % (FTe_ssim_mean)
                with open('RD_AFR_SSIM.txt', 'a') as f:
                    f.write(line_SSIM)
                #metric_summary = tf.Summary(value=[tf.Summary.Value(tag='FTE_PSNR', simple_value=FTe_psnr_mean),tf.Summary.Value(tag='FTE_SSIM', simple_value=FTe_ssim_mean)])
                #self.writer.add_summary(summary=metric_summary, global_step=epoch+1)
            ############################PSNR and SSIM################################################
                ######Record LR_Change##############
                line2 = "Epoch:[%3d], LR: %.6f \n" % (epoch+1, curr_lr)
                with open('LR_REC.txt', 'a') as f:
                    f.write(line2)
                ######Record LR_Change##############
                if FTe_ssim_mean > 0.9316:
                    self.save(self.checkpoint_dir, counter)
            epoch += 1
            iterations = 0

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.gan_type)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def test(self):
        print('---------------------------')
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)
        save_img = True
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_rainy_path = '../../Derain_Data/test_a/data/'
        test_gt_path = '../../Derain_Data/test_a/gt/'

        FTe_psnr = []
        FTe_ssim = []
        for Fsample_ID in range(58):
            FTe_batch_r, FTe_batch_g, RZero, BZero = read_tridata(test_rainy_path, test_gt_path, self.Train_W, self.Train_H, Fsample_ID, 1)
            #print(FTe_batch_r.shape)
            FTe_derain_img = self.sess.run(self.TFU_DR, feed_dict={self.TestR: FTe_batch_r,self.TestG: FTe_batch_g})
            #FRA_imgs = FTe_batch_r[0,:,:,:]
            #FRA_imgs = ((FRA_imgs+1)*127.5).astype(np.uint8)
            FGT_imgs = FTe_batch_g[0,:,:,:]
            FGT_imgs = ((FGT_imgs+1.0)*127.5).astype(np.uint8)
            FDerain_imgs = FTe_derain_img[0,:,:,:]
            FDerain_imgs = post_ouput(FDerain_imgs)
            FDerain_imgs = ((FDerain_imgs+1.0)*127.5).astype(np.uint8)
            #F_Masks = FTe_mask[0,:,:,:]
            #F_Masks = ((F_Masks+1)*127.5).astype(np.uint8)
            RZero_imgs = RZero[0,:,:,:]
            RZero_imgs = ((RZero_imgs)*255.0).astype(np.uint8)
            BZero_imgs = BZero[0,:,:,:]
            BZero_imgs = ((BZero_imgs)*255.0).astype(np.uint8)
            Fcur_psnr = calc_psnr(FGT_imgs, FDerain_imgs)
            Fcur_ssim = calc_ssim(FGT_imgs, FDerain_imgs)
            #print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            #print('Current iteration for calculating metric is: %d'%(i_PS))
            #Tr_psnr += trcur_psnr
            FTe_psnr.append(Fcur_psnr)
            FTe_ssim.append(Fcur_ssim)
            if save_img:
                if Fsample_ID >= 0 and Fsample_ID <= 9:
                    #cv2.imwrite(result_dir+'/' + 'Rainy' + '_00%d.png'%Fsample_ID, FRA_imgs)
                    cv2.imwrite(result_dir+'/' + 'Clean' + '_00%d.png'%Fsample_ID, FGT_imgs)
                    cv2.imwrite(result_dir+'/' + 'Derain' + '_00%d.png'%Fsample_ID, FDerain_imgs)
                    #cv2.imwrite(result_dir+'/' + 'RZero' + '_00%d.png'%Fsample_ID, RZero_imgs)
                    #cv2.imwrite(result_dir+'/' + 'BZero' + '_00%d.png'%Fsample_ID, BZero_imgs)
                if Fsample_ID >= 10:
                    #cv2.imwrite(result_dir+'/' + 'Rainy' + '_0%d.png'%Fsample_ID, FRA_imgs)
                    cv2.imwrite(result_dir+'/' + 'Clean' + '_0%d.png'%Fsample_ID, FGT_imgs)
                    cv2.imwrite(result_dir+'/' + 'Derain' + '_0%d.png'%Fsample_ID, FDerain_imgs)
                    #cv2.imwrite(result_dir+'/' + 'RZero' + '_0%d.png'%Fsample_ID, RZero_imgs)
                    #cv2.imwrite(result_dir+'/' + 'BZero' + '_0%d.png'%Fsample_ID, BZero_imgs)
        FTe_psnr_mean = np.mean(FTe_psnr)
        FTe_ssim_mean = np.mean(FTe_ssim)
        FTe_psnr_max = np.max(FTe_psnr)
        FTe_ssim_max = np.max(FTe_ssim)
        FTe_psnr_min = np.min(FTe_psnr)
        FTe_ssim_min = np.min(FTe_ssim)
        FTE_PSNR = np.array(FTe_psnr)
        FTE_SSIM = np.array(FTe_ssim)
        np.savetxt(self.epoch_metric_dir+'/'+'FTE_PSNR.txt',FTE_PSNR)
        np.savetxt(self.epoch_metric_dir+'/'+'FTE_SSIM.txt',FTE_SSIM)
        print('==============================')
        print('Sample number for PSNR/SSIM calculation is: %d'%(Fsample_ID+1))
        print('PSNR is %.4f (%.4f/%.4f) and SSIM is %.4f (%.4f/%.4f)'%(FTe_psnr_mean, FTe_psnr_max, FTe_psnr_min, FTe_ssim_mean, FTe_ssim_max, FTe_ssim_min))
        print('==============================')
