from ops import *
from utils import *
from DrawUtils import *
from random import shuffle
import tensorflow as tf
import scipy.io as sio


class Model(object):
    def __init__(self, sess, trainRoot='./data', testRoot='./data', rstRoot= './data', batchSize = 5):
        self.sess = sess
        self.trainRoot = trainRoot
        self.testRoot = testRoot
        self.rstRoot = rstRoot
        self.batchSize = batchSize
        self.ImgH = 1024
        self.ImgW = 1024
        self.vH = 128
        self.vW = 128
        self.vD = 96
        self.lam = 10
        self.Gen_learningRate = 0.0001
        self.Dis_learningRate = 0.0003
        self.sliceShowID = 46
        self.critic_iteration = 1

        self.build_model()

    def save(self, checkpoint_dir, step):
        model_name = "GAN_HairStruc.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def attention3D(self, x, ch, dch=8, sn=True, name='attention'):
        f = conv3d(x, ch // dch, k_d=1, k_h=1, k_w=1, d_d=1, d_h=1, d_w=1, sn=sn, name=name+'f_conv')  # [bs, h, w, c']
        g = conv3d(x, ch // dch, k_d=1, k_h=1, k_w=1, d_d=1, d_h=1, d_w=1, sn=sn, name=name+'g_conv')  # [bs, h, w, c']
        h = conv3d(x, ch, k_d=1, k_h=1, k_w=1, d_d=1, d_h=1, d_w=1, sn=sn, name=name+'h_conv')  # [bs, h, w, c]
        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable(name+"gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.get_shape().as_list())  # [bs, h, w, C]
        x = gamma * o + x
        return x

    def Generator_V(self, Img, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            sn = False
            '''1024 x 1024 x 4 --> 512 x 512 x 16'''
            h0 = tf.nn.relu(conv2d(Img, 16, sn=sn, name='g_vh0_conv'))
            h00 = tf.nn.relu(conv2d(Img, 8, sn=sn, name='g_vh00_conv'))
            h00 = tf.nn.relu(conv2d(h00, 16, d_h=1, d_w=1, sn=sn, name='g_vh01_conv'))
            h0 = tf.nn.relu(h0+h00)
            '''512 x 512 x 16 --> 256 x 256 x 64'''
            h1 = tf.nn.relu(conv2d(h0, 64, sn=sn, name='g_vh1_conv'))
            h10 = tf.nn.relu(conv2d(h0, 32, sn=sn, name='g_vh10_conv'))
            h10 = tf.nn.relu(conv2d(h10, 64, d_h=1, d_w=1, sn=sn, name='g_vh11_conv'))
            h1 = tf.nn.relu(h1+h10)
            '''256 x 256 x 64 --> 128 x 128 x 256'''
            h2 = tf.nn.relu(conv2d(h1, 256, sn=sn, name='g_vh2_conv'))
            h20 = tf.nn.relu(conv2d(h1, 128, sn=sn, name='g_vh20_conv'))
            h20 = tf.nn.relu(conv2d(h20, 256, d_h=1, d_w=1, sn=sn, name='g_vh21_conv'))
            h2 = tf.nn.relu(h2+h20)
            '''128 x 128 x 256'''
            h30 = tf.nn.relu(conv2d(h2, 256, d_h=1, d_w=1, sn=sn, name='g_vh30_conv'))
            h30 = tf.nn.relu(conv2d(h30, 256, d_h=1, d_w=1, sn=sn, name='g_vh31_conv'))
            h3 = tf.nn.relu(h30 + h2)
            return h3

    def Generator_X(self, x, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            sn = False
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(x, 256, d_h=1, d_w=1, sn=sn, name='g_xh00_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_xh01_conv'))
            h0 = tf.nn.relu(h00+x)
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(h0, 256, d_h=1, d_w=1, sn=sn, name='g_xh10_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_xh11_conv'))
            h0 = tf.nn.relu(h00 + h0)
            '''128 x 128 x 256 --> 128 x 128 x 128'''
            h0 = tf.nn.relu(conv2d(h0, 128, d_h=1, d_w=1, sn=sn, name='g_xh2_conv'))
            '''128 x 128 x 128 --> 128 x 128 x 96'''
            h0 = tf.nn.relu(conv2d(h0, 96, d_h=1, d_w=1, sn=sn, name='g_xh3_conv'))

            '''trans 2D to 3D'''
            h0 = tf.expand_dims(h0, 4)
            return h0

    def Generator_Y(self, y, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            sn = False
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(y, 256, d_h=1, d_w=1, sn=sn, name='g_yh00_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_yh01_conv'))
            h0 = tf.nn.relu(h00 + y)
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(h0, 256, d_h=1, d_w=1, sn=sn, name='g_yh10_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_yh11_conv'))
            h0 = tf.nn.relu(h00 + h0)
            '''128 x 128 x 256 --> 128 x 128 x 128'''
            h0 = tf.nn.relu(conv2d(h0, 128, d_h=1, d_w=1, sn=sn, name='g_yh2_conv'))
            '''128 x 128 x 128 --> 128 x 128 x 96'''
            h0 = tf.nn.relu(conv2d(h0, 96, d_h=1, d_w=1, sn=sn, name='g_yh3_conv'))

            '''trans 2D to 3D'''
            h0 = tf.expand_dims(h0, 4)
            return h0

    def Generator_Z(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            sn = False
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(z, 256, d_h=1, d_w=1, sn=sn, name='g_zh00_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_zh01_conv'))
            h0 = tf.nn.relu(h00 + z)
            '''128 x 128 x 256'''
            h00 = tf.nn.relu(conv2d(h0, 256, d_h=1, d_w=1, sn=sn, name='g_zh10_conv'))
            h00 = tf.nn.relu(conv2d(h00, 256, d_h=1, d_w=1, sn=sn, name='g_zh11_conv'))
            h0 = tf.nn.relu(h00 + h0)
            '''128 x 128 x 256 --> 128 x 128 x 128'''
            h0 = tf.nn.relu(conv2d(h0, 128, d_h=1, d_w=1, sn=sn, name='g_zh2_conv'))
            '''128 x 128 x 128 --> 128 x 128 x 96'''
            h0 = tf.nn.relu(conv2d(h0, 96, d_h=1, d_w=1, sn=sn, name='g_zh3_conv'))

            '''trans 2D to 3D'''
            h0 = tf.expand_dims(h0, 4)
            return h0

    def Generator_A(self, G, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            sn = False
            h = tf.nn.relu(conv3d(G, 3, d_d=1, d_h=1, d_w=1, sn=sn, name='g_Ah00_conv3D'))
            h = tf.nn.relu(conv3d(h, 3, d_d=1, d_h=1, d_w=1, sn=sn, name='g_Ah01_conv3D'))
            G = tf.nn.relu(h + G)
            h = tf.nn.relu(conv3d(G, 3, d_d=1, d_h=1, d_w=1, sn=sn, name='g_Ah10_conv3D'))
            h = tf.nn.relu(conv3d(h, 3, d_d=1, d_h=1, d_w=1, sn=sn, name='g_Ah11_conv3D'))
            G = tf.nn.relu(h + G)
            return G

    def Generator(self, Img, reuse=False):
        V = self.Generator_V(Img, reuse)
        X = self.Generator_X(V, reuse)
        Y = self.Generator_Y(V, reuse)
        Z = self.Generator_Z(V, reuse)
        G = tf.concat([X, Y, Z], axis=4)
        G = self.Generator_A(G, reuse)
        return G

    def Discriminator_I(self, Img, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            '''1024 x 1024 x 4 --> 512 x 512 x 32'''
            h0 = lrelu(conv2d(Img, 32, name='d_ih0_conv'))
            '''512 x 512 x 32 --> 256 x 256 x 64'''
            h0 = lrelu(conv2d(h0, 64, name='d_ih1_conv'))
            '''256 x 256 x 64 --> 128 x 128 x 128'''
            h0 = lrelu(conv2d(h0, 128, name='d_ih2_conv'))
            '''128 x 128 x 128 --> 128 x 128 x 96'''
            h0 = lrelu(conv2d(h0, 96, d_w=1, d_h=1, name='d_ih3_conv'))
            h0 = tf.expand_dims(h0, 4)
            return h0

    def Discriminator_V(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            fd = 32
            '''[128 x 128 x 96] x 4 --> [64 x 64 x 48] x 32'''
            h0 = lrelu(conv3d(x, fd, name='d_vh0_conv'))
            '''[64 x 64 x 48] x 32 --> [32 x 32 x 24] x 64'''
            h1 = lrelu(conv3d(h0, fd*2, name='d_vh1_conv'))
            '''[32 x 32 x 24] x 64 --> [16 x 16 x 12] x 128'''
            h2 = lrelu(conv3d(h1, fd*4, name='d_vh2_conv'))

            '''[16 x 16 x 12] x 128 --> [8 x 8 x 6] x 256'''
            h3 = lrelu(conv3d(h2, fd*8, name='d_vh3_conv'))
            '''[8 x 8 x 6] x 256 --> [4 x 4 x 3] x 512'''
            d = lrelu(conv3d(h3, fd * 16, name='d_vh4_conv'))

            d = tf.reshape(d, [self.batchSize, -1])
            d = linear(d, 1, 'd_vh5_lin')
            return d, h3, h2, h1, h0

    def content_layer_loss(self, p, x):
        b, h, w, d, c = p.get_shape()
        K = 1./(2.*b.value)
        loss = K*tf.reduce_sum(tf.pow(p-x, 2))
        return loss

    def gram_matrix(self, x, bsize, area, depth):
        F = tf.reshape(x, [bsize, area, depth])
        TF = tf.transpose(F, perm=[0, 2, 1])
        G = tf.matmul(TF, F)
        return G

    def style_layer_loss(self, a, x):
        b, h, w, d, c = a.get_shape()
        M = h.value*w.value*d.value
        N = c.value
        A = self.gram_matrix(a, b.value, M, N)
        print(A.get_shape())
        X = self.gram_matrix(x, b.value, M, N)
        loss = (1./(4*N**2*M**2))*tf.reduce_sum(tf.pow((X-A), 2))
        return loss/b.value


    def build_model(self):
        Img_dims = [self.ImgH, self.ImgW, 4]
        Vol_dims = [self.vH, self.vW, self.vD, 3]

        self.ImgIn  = tf.placeholder(tf.float32, [self.batchSize]+Img_dims, name='ImgIn')
        self.VolGT = tf.placeholder(tf.float32, [self.batchSize]+Vol_dims, name='VolOut')

        '''Model output'''
        self.G = self.Generator(self.ImgIn, reuse=False)

        '''Opt Energy'''
        ixx = self.Discriminator_I(self.ImgIn, reuse=False)
        iD_real = tf.concat([self.VolGT, ixx], 4)
        self.D_real, dr_f3, dr_f2, dr_f1, dr_f0 = self.Discriminator_V(iD_real, reuse=False)
        iD_fake = tf.concat([self.G, ixx], 4)
        self.D_fake, df_f3, df_f2, df_f1, df_f0 = self.Discriminator_V(iD_fake, reuse=True)

        '''D_loss'''
        eps = tf.random_uniform([self.batchSize, 1, 1, 1, 1], minval=0., maxval=1.)
        X_inter = eps * self.VolGT + (1. - eps) * self.G
        X_inter = tf.concat([X_inter, ixx], 4)
        DX_inter, _, _, _,_ = self.Discriminator_V(X_inter, reuse=True)
        gradv = tf.gradients(DX_inter, [X_inter])[0]
        gradv = tf.reshape(gradv, [self.batchSize, -1])
        grad_norm = tf.reduce_sum(gradv**2, axis=1)
        grad_norm = tf.sqrt(grad_norm)
        grad_pen = self.lam * tf.reduce_mean((grad_norm - 1.)**2)
        self.D_loss = tf.reduce_mean(self.D_fake)-tf.reduce_mean(self.D_real) + grad_pen

        '''G_loss'''
        self.fStyleLoss = self.style_layer_loss(df_f3, dr_f3) + self.style_layer_loss(df_f2, dr_f2) \
                          + self.style_layer_loss(df_f1, dr_f1) + self.style_layer_loss(df_f0, dr_f0) \
                          + self.style_layer_loss(self.G, self.VolGT)
        self.featureLoss = 0.5 * self.content_layer_loss(self.G, self.VolGT) + 0.5 * self.content_layer_loss(df_f2, dr_f2)
        # self.featureLoss = tf.reduce_sum(tf.pow((df_f3-dr_f3), 2)) +\
        #                    tf.reduce_sum(tf.pow((df_f2-dr_f2), 2)) + tf.reduce_mean(tf.pow(df_f1-dr_f1, 2))
        # self.featureLoss = tf.reduce_sum(tf.abs(self.VolGT - self.G)) / (self.batchSize * self.vW * self.vH * self.vD)

        KWeiFContent_alpha = 1.e-2
        KWeiSContent_beta = 5.e+2 * KWeiFContent_alpha
        # self.G_loss = -tf.reduce_mean(self.D_fake) + self.featureLoss
        self.G_loss = KWeiFContent_alpha * self.featureLoss + KWeiSContent_beta * self.fStyleLoss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

    def get_ShufferTrainBatch(self, idx, trainList, epoSize):
        if idx % epoSize == 0:
            shuffle(trainList)
        offset = (idx % epoSize) * self.batchSize
        tImgInfo, tVoxGT = getTrainBatch(trainList, offset, self.batchSize)
        return tImgInfo, tVoxGT, idx+1

    def test(self, inDir):
        loadFlag, cpt_counter = self.load(self.rstRoot + "/checkpoint/")
        if not loadFlag:
            print("No parameters!!")
            return
        import time
        start_time = time.time()
        inList = get_fileSet_list(inDir)
        inputInfo = loadImgInput(inList)
        inputInfo = np.resize(inputInfo, [1, self.ImgH, self.ImgW, 4])
        testInput = tf.placeholder(tf.float32, [1, self.ImgH, self.ImgW, 4])
        testG = self.Generator(testInput, reuse=True)
        tG = self.sess.run([testG], feed_dict={testInput: inputInfo})
        tG = np.array(tG[0])
        tGTShow = ImgMergeVoxSlice(tG, 1, self.sliceShowID)
        cv2.imwrite(inDir+"TShow.png", tGTShow)
        tG = voxelMatTrans(tG)
        sio.savemat(inDir+"Ori_out.mat", {'Ori': tG})
        print("--- %s seconds ---" % (time.time() - start_time))

    def interHairTest(self, inDir, inDirA, inDirB):
        loadFlag, cpt_counter = self.load(self.rstRoot + "/checkpoint/")
        if not loadFlag:
            print("No parameters!!")
            return
        A_inList = get_fileSet_list(inDirA)
        A_inputInfo = loadImgInput(A_inList)
        A_inputInfo = np.resize(A_inputInfo, [1, self.ImgH, self.ImgW, 4])

        B_inList = get_fileSet_list(inDirB)
        B_inputInfo = loadImgInput(B_inList)
        B_inputInfo = np.resize(B_inputInfo, [1, self.ImgH, self.ImgW, 4])

        A_In = tf.placeholder(tf.float32, [1, self.ImgH, self.ImgW, 4])
        B_In = tf.placeholder(tf.float32, [1, self.ImgH, self.ImgW, 4])

        A_V = self.Generator_V(A_In, reuse=True)
        B_V = self.Generator_V(B_In, reuse=True)
        cV = 0.5 * A_V + (1. - 0.5) * B_V
        cX = self.Generator_X(cV, reuse=True)
        cY = self.Generator_Y(cV, reuse=True)
        cZ = self.Generator_Z(cV, reuse=True)
        cG = tf.concat([cX, cY, cZ], axis=4)
        cG = self.Generator_A(cG, reuse=True)

        tG = self.sess.run([cG], feed_dict={A_In: A_inputInfo, B_In: B_inputInfo})
        tG = np.array(tG[0])
        tGTShow = ImgMergeVoxSlice(tG, 1, self.sliceShowID)
        cv2.imwrite(inDir + "TShow.png", tGTShow)
        tG = voxelMatTrans(tG)
        sio.savemat(inDir + "Ori_out.mat", {'Ori': tG})


    def train(self, maxIter):
        D_solver = tf.train.AdamOptimizer(learning_rate=self.Dis_learningRate, beta1=0., beta2=0.9)\
            .minimize(self.D_loss, var_list=self.d_vars)
        G_solver = tf.train.AdamOptimizer(learning_rate=self.Gen_learningRate, beta1=0., beta2=0.9)\
            .minimize(self.G_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        testList = get_fileSet_list(self.testRoot)
        testII, testOO = getTrainBatch(testList, 0, self.batchSize)
        tGTShow = ImgMergeVoxSlice(testOO, self.batchSize, self.sliceShowID)
        cv2.imwrite(self.rstRoot+"/GT.png", tGTShow)

        cv2.imwrite(self.rstRoot + "/c_0.png", testII[0, :, :, 0]*255.)
        cv2.imwrite(self.rstRoot + "/c_1.png", testII[0, :, :, 1]*255.)
        cv2.imwrite(self.rstRoot + "/c_2.png", testII[0, :, :, 2] * 255.)
        cv2.imwrite(self.rstRoot + "/c_3.png", testII[0, :, :, 3] * 255.)

        trainList = get_fileSet_list(self.trainRoot)
        epoSize = len(trainList) // self.batchSize

        loadFlag, cpt_counter = self.load(self.rstRoot+"/checkpoint/")
        BegI = cpt_counter+1 if loadFlag else 0

        idx = 0
        self.ss_dloss = []
        self.ss_gloss = []
        self.ss_tdloss = []
        self.ss_tdcout = []
        for itter in range(BegI, maxIter):
            for c in range(self.critic_iteration):
                cImg, cVox, idx = self.get_ShufferTrainBatch(idx, trainList, epoSize)
                _, _Dloss = self.sess.run([D_solver, self.D_loss], feed_dict={self.ImgIn: cImg, self.VolGT: cVox})
                print("(%d, %d): D_loss: %.8f" % (itter, c, _Dloss))

            cImg, cVox, idx = self.get_ShufferTrainBatch(idx, trainList, epoSize)
            _, _Gloss, _Dloss, _ffLoss, _fSLoss = self.sess.run([G_solver, self.G_loss, self.D_loss,
                                                                 self.featureLoss, self.fStyleLoss],
                                                                feed_dict={self.ImgIn: cImg, self.VolGT: cVox})
            print("Iter_%d: D_loss: %.8f, G_loss: %.8f, ff_loss: %.8f, fs_loss: %.8f" % (itter, _Dloss, _Gloss, _ffLoss,
                                                                                         _fSLoss))
            self.ss_dloss.append(-_Dloss)
            self.ss_gloss.append(_Gloss)

            if itter % 100 == 0 or itter == maxIter-1:
                testG, _Dloss, _Gloss = self.sess.run([self.G, self.D_loss, self.G_loss],
                                                      feed_dict={self.ImgIn: testII, self.VolGT: testOO})
                self.ss_tdloss.append(-_Dloss)
                self.ss_tdcout.append(itter)
                testG = np.array(testG)
                tGTShow = ImgMergeVoxSlice(testG, self.batchSize, self.sliceShowID)
                cv2.imwrite(self.rstRoot + "/t_%d.png" % itter, tGTShow)
                self.save(self.rstRoot+"/checkpoint/", itter)

                dloss = np.array(self.ss_dloss)
                gloss = np.array(self.ss_gloss)
                tdloss = np.array(self.ss_tdloss)
                tdCount = np.array(self.ss_tdcout)
                if not os.path.exists(self.rstRoot + "/Loss/"):
                    os.makedirs(self.rstRoot + "/Loss/")
                sio.savemat(self.rstRoot + "/Loss/loss_%d.mat" % itter, {'dl': dloss, 'gl': gloss,
                                                                  'tdl': tdloss, 'tdc': tdCount})

                self.ss_dloss.clear()
                self.ss_gloss.clear()
                self.ss_tdloss.clear()
                self.ss_tdcout.clear()
















