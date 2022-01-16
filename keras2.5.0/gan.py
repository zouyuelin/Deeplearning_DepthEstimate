import argparse
from inspect import classify_class_attrs
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2 as cv
import os
import time

#采用静态图的形式，可关闭急切模式
#tf.compat.v1.disable_eager_execution() 

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--type',default='gp')
args = parser.parse_args()

class datasets:
    def __init__(self, datasetsPath:str,type='gp'):
        self.dataPath = datasetsPath
        self.type = type
        self.noise_dim = 128
        self.dim = 64
        self.epochs = 400
        self.batch_size = 32
        self.data_num = 12500
        self.learning_rate = 2e-4
        self.save_step = 300
        self.n_critic = 1
        self.n_generate = 1
        self.save_path = 'generateImage/'
        self.model_path = 'checkpoints/'

        self.classifyImages()
        self.buildTrainData()

    def classifyImages(self):
        imageList = os.listdir(self.dataPath)
        np.random.seed(10101)
        np.random.shuffle(imageList)
        self.catImages = []
        self.dogImages = []
        for index in imageList:
            if 'cat' in index:
                self.catImages.append(os.path.join(self.dataPath,index))
            elif 'dog' in index:
                self.dogImages.append(os.path.join(self.dataPath,index))

    def load_image(self,imagePath:tf.Tensor):
        img = img = tf.io.read_file(imagePath)
        img = tf.image.decode_jpeg(img) #此处为jpeg格式
        img = tf.image.resize(img,(self.dim,self.dim))/255.0
        #img = tf.reshape(img,[self.dim,self.dim,3])
        img = tf.cast(img,tf.float32)
        return img

    def buildTrainData(self):
        '''
        you can watch the datasets use function take;\\
        For example:
            img = traindata.ds_train.take(3)\\
            print(np.shape(np.array(list(img.as_numpy_iterator())))) #(3, 32, 64, 64, 3)

            for img in traindata.ds_train.take(3):\\
                print(img)\\
                image = np.array(img[0]*255,np.uint8)\\
                cv.imshow("asf",image)\\
                cv.waitKey(0)\\
        ''' 
        self.ds_train = tf.data.Dataset.from_tensor_slices(self.catImages) \
           .map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
           .repeat(1000) \
           .shuffle(buffer_size = 500).batch(self.batch_size) \
           .prefetch(tf.data.experimental.AUTOTUNE).cache()  
        
        self.itertor_train = iter(self.ds_train)

#-------------------------------------------------------------------
#                            define resBlock                       |
#-------------------------------------------------------------------   

class resBlock(K.layers.Layer):
    def __init__(self,num_filters, resampling,strides=2, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.resampling = resampling
        self.strides = strides

    def build(self, input_shape):
        self.upsampl1 = K.layers.UpSampling2D()
        self.covC1 = K.layers.Conv2D(self.num_filters, kernel_size=(3,3), strides=1, padding='same',
                        kernel_initializer='he_normal')
        self.covC1_1 = K.layers.Conv2D(self.num_filters, kernel_size=(3,3), strides=self.strides,  padding='same',
                        kernel_initializer='he_normal')

        self.BN = K.layers.BatchNormalization()
        self.relu = K.layers.Activation('relu')
        self.LeakyRelu = K.layers.LeakyReLU()

        self.cov1 = K.layers.Conv2D(self.num_filters, kernel_size=(3,3), strides=1,padding='same',
                        kernel_initializer='he_normal')

        self.BN1 = K.layers.BatchNormalization()

        self.cov2 = K.layers.Conv2D(self.num_filters, kernel_size=(3,3), strides=1,padding='same',
                        kernel_initializer='he_normal')

        self.BN2 = K.layers.BatchNormalization()

        self.upsampl2 = K.layers.UpSampling2D()
        self.covC2 = K.layers.Conv2D(self.num_filters, kernel_size=(1,1), strides=1, padding='same',
                        kernel_initializer='he_normal')
        self.covC2_1 = K.layers.Conv2D(self.num_filters, kernel_size=(1,1), strides=self.strides,  padding='same',
                        kernel_initializer='he_normal')

        self.BN3 = K.layers.BatchNormalization()

        self.add = K.layers.Add()
        return super().build(input_shape)

    def call(self, inputs):
        #F1,F2,F3 = num_filters
        X_shortcut = inputs
        
        #//up or down
        x = K.layers.Lambda(lambda x: x)(inputs)
        if self.resampling is 'up':
            x = self.upsampl1(x)
            x = self.covC1(x)
            #x = keras.layers.Conv2DTranspose(num_filters,kernel_size=kernel_size, strides=2,  padding='same',
            #              kernel_initializer=keras.initializers.RandomNormal())(x)
        elif self.resampling is 'down':
            x = self.covC1_1(x)
        
        #//BN_relu
        x = self.BN(x)
        x = self.relu(x)
        #x = self.LeakyRelu(x)

        #//cov2d
        x = self.cov1(x)
        
        #//BN_relu
        x = self.BN1(x)
        x = self.relu(x)
        #x = self.LeakyRelu(x)
        
        #//cov2d
        x = self.cov2(x)
        #//BN_relu
        x = self.BN2(x)
        
        #//add_shortcut
        if self.resampling is 'up':
            X_shortcut = self.upsampl2(X_shortcut)
            X_shortcut = self.covC2(X_shortcut)
            #x = keras.layers.Conv2DTranspose(num_filters,kernel_size=kernel_size, strides=2,  padding='same',
            #              kernel_initializer=keras.initializers.RandomNormal())(x)
        elif self.resampling is 'down':
            X_shortcut = self.covC2_1(X_shortcut)
        X_shortcut = self.BN3(X_shortcut)
        
        X_add = self.add([x,X_shortcut])
        X_add = self.relu(X_add)
        #X_add = self.LeakyRelu(X_add)
        
        return X_add


class generator(K.layers.Layer):
    def __init__(self,resampling='up', **kwargs):
        super(generator,self).__init__(**kwargs)
        self.resampling = resampling

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'resampling': self.resampling
        })
        return config
        
    def build(self, input_shape):
        self.dense_1 = K.layers.Dense(512*4*4)
        self.reshape_1 = K.layers.Reshape((4,4,512))
        self.BN = K.layers.BatchNormalization()
        self.relu = K.layers.Activation('relu')
        self.LeakyRelu = K.layers.LeakyReLU()

        self.resblock_1 = resBlock(num_filters=512,resampling=self.resampling)
        self.resblock_2 = resBlock(num_filters=256,resampling=self.resampling)
        self.resblock_3 = resBlock(num_filters=128,resampling=self.resampling)
        self.resblock_4 = resBlock(num_filters=64,resampling=self.resampling)

        self.conv2d = K.layers.Conv2D(3, kernel_size=(3,3), strides=1, padding='same',
                        kernel_initializer='he_normal')
        self.tanh = K.layers.Activation('tanh')

        return super().build(input_shape)

    def call(self, inputs):
        g = self.dense_1(inputs)
        g = self.reshape_1(g)
        #//BN_relu
        g = self.BN(g)
        g = self.relu(g)
        #g = self.LeakyRelu(g)
        #4*4*512
        g = self.resblock_1(g)
        #8*8*512
        g = self.resblock_2(g)
        #16*16*256
        g = self.resblock_3(g)
        #32*32*128
        g = self.resblock_4(g)
        #64*64*64
        g = self.conv2d(g)
        #64*64*3
        g_out = self.tanh(g)
        return g_out

class discriminator(K.layers.Layer):
    def __init__(self,resampling='down', **kwargs):
        super(discriminator,self).__init__(**kwargs)
        self.resampling = resampling

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'resampling': self.resampling
        })
        return config

    def build(self, input_shape):
        self.conv2d = K.layers.Conv2D(64, kernel_size=(3,3), padding='same',strides=1,
                        kernel_initializer='he_normal')
        self.BN = K.layers.BatchNormalization()
        self.relu = K.layers.Activation('relu')
        self.LeakyRelu = K.layers.LeakyReLU()

        self.resblock_1 = resBlock(num_filters=128,resampling=self.resampling)
        self.resblock_2 = resBlock(num_filters=256,resampling=self.resampling)
        self.resblock_3 = resBlock(num_filters=512,resampling=self.resampling)
        self.resblock_4 = resBlock(num_filters=512,resampling=self.resampling)


        self.averagePool2d = K.layers.GlobalAveragePooling2D()
        self.dense = K.layers.Dense(1)
        return super().build(input_shape)

    def call(self, inputs):
        d = self.conv2d(inputs)
        #//BN_relu
        d = self.BN(d)
        d = self.relu(d)
        #d = self.LeakyRelu(d)
        #64*64*64
        d = self.resblock_1(d) 
        #32*32*128
        d = self.resblock_2(d)
        #16*16*256
        d = self.resblock_3(d)
        #8*8*512
        d = self.resblock_4(d)
        #4*4*512
        '''
            GlobalAveragePooling :it can replace the full connection layer
            you can use the Dense to test the network
        '''
        d = self.averagePool2d(d)
        d_out = self.dense(d)
        return d_out

class wgan:
    def __init__(self,datasets:datasets):
        self.traindata = datasets
    def build(self):
        assert self.traindata.type == 'gp' or self.traindata.type == 'div', f'please confirm the type is {self.traindata.type}'
        #------------------------------
        #define the generate model    *
        #------------------------------
        self.generate_model = K.Sequential()
        self.generate_model.add(generator())
        #--------------------------------
        #define the discriminator model *
        #--------------------------------
        self.discriminator_model = K.Sequential()
        self.discriminator_model.add(discriminator())

        #--------------------------------
        #      combine the model        *
        #--------------------------------
        z_noise = K.layers.Input(shape=(self.traindata.noise_dim,))
        #get the score
        fake_img = self.generate_model(z_noise)
        fake_score = self.discriminator_model(fake_img)
        self.combineModel = K.Model(z_noise,fake_score)

        #--------------------------------
        #        optimizer              *
        #--------------------------------
        self.discriminator_optimizer=K.optimizers.Adam(self.traindata.learning_rate, 0.5)
        self.generator_optimizer=K.optimizers.Adam(self.traindata.learning_rate, 0.5)
        # self.generator_optimizer.lr

        self.generate_model.summary()
        self.discriminator_model.summary()
        self.combineModel.summary()

    def train_discriminator(self,z_noise,train_data,u_niform):
        k = 2
        p = 6
        u = u_niform
        with tf.GradientTape() as tape,\
            tf.GradientTape() as d_tape:
            D_fake_img = self.generate_model(z_noise)
            D_fake_score = self.discriminator_model(D_fake_img)
            D_real_score = self.discriminator_model(train_data)
            #get a new input consist from fake and real
            x_ = (1.-u)*train_data+u*D_fake_img
            #//
            #-------------------------------------------------------------------
            #                            wgan div loss function                |
            #                               n_critic = 1                       |
            #                          arxiv.org/pdf/1712.01026.pdf            |
            #-------------------------------------------------------------------
            #//
            if self.traindata.type == 'div':
                gradients = tape.gradient(self.discriminator_model(x_), [x_])[0]
                grad_norm = K.backend.sqrt(K.backend.sum(gradients ** 2, axis=[1, 2, 3])) ** p
                grad_penalty = k * K.backend.mean(grad_norm)
                discriminator_loss = K.backend.mean(D_real_score - D_fake_score)
            #//
            #-------------------------------------------------------------------
            #                            wgan gp  loss function                |
            #                               n_critic = 5                       |
            #                          arxiv.org/pdf/1704.00028.pdf            |
            #-------------------------------------------------------------------
            #//
            elif self.traindata.type == 'gp':
                gradients = tape.gradient(self.discriminator_model(x_), [x_])[0]
                grad_norm = K.backend.sqrt(K.backend.sum(gradients ** 2, axis=[1, 2, 3]))
                grad_norm = K.backend.square(1-grad_norm)
                grad_penalty =  10*K.backend.mean(grad_norm)
                discriminator_loss = K.backend.mean(D_fake_score-D_real_score)
            discriminator_loss_all = grad_penalty + discriminator_loss
        gradients_d = d_tape.gradient(discriminator_loss_all,self.discriminator_model.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_d,self.discriminator_model.trainable_variables))
        return discriminator_loss_all,gradients_d

    def train_generator(self,z_noise,train_data):
        with tf.GradientTape() as g_tape:
            G_fake_img = self.generate_model(z_noise)
            G_fake_score = self.discriminator_model(G_fake_img)
            #G_real_score = self.discriminator_model(train_data)
            if self.traindata.type == 'div':
                generate_loss = K.backend.mean(G_fake_score)
                #generate_loss = K.backend.mean(K.backend.square(G_fake_score-G_real_score))
            if self.traindata.type == 'gp':
                generate_loss = -K.backend.mean(G_fake_score)#min this value
                #generate_loss = K.backend.mean(K.backend.square(G_fake_score-G_real_score))
        gradients_g = g_tape.gradient(generate_loss,self.generate_model.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_g,self.generate_model.trainable_variables))
        return generate_loss,gradients_g

    def train(self):
        if os.path.exists(os.path.join(self.traindata.model_path,'gan.h5')):
            self.combineModel.load_weights(os.path.join(self.traindata.model_path,'gan.h5'))
            if os.path.exists(os.path.join(self.traindata.model_path,'history.npy')):
                history = np.load(os.path.join(self.traindata.model_path,'history.npy'), allow_pickle=True).tolist()
                #read the last data use -1 index,and use 0 to read the first data
                #\\
                last_iter = int(history[-1][0])
                print('Find the npy file, the last save iter:%d' % (last_iter))
            else:
                history = []
                last_iter = -1
        else:
            print('There is no .npy file, creating a new file---------')
            history = []
            last_iter = -1


        for step in range(last_iter+1,int(self.traindata.epochs*self.traindata.data_num/self.traindata.batch_size+1)):
            try:
                #get the time
                start_time = time.time()

                train_data = self.traindata.itertor_train.get_next()
                z_noise = np.random.normal(size=self.traindata.batch_size*self.traindata.noise_dim)\
                                        .reshape([self.traindata.batch_size,self.traindata.noise_dim])
                u_niform = np.random.uniform(low=0.0,high=1.0,size=(self.traindata.batch_size,1,1,1))

                # training the model
                for i in range(self.traindata.n_critic):
                    discriminator_loss_all,gradients_d = self.train_discriminator(z_noise,train_data,u_niform)
                for i in range(self.traindata.n_generate):
                    generate_loss,gradients_g = self.train_generator(z_noise,train_data)
                

                #get the time 
                duration = time.time()-start_time
                
                #-----------------------------------------
                #            print the loss              |
                #-----------------------------------------
                if step % 5 == 0:
                    tf.print("The step is %s,d_loss:%s,g_loss:%s"%(step,
                                        np.array(discriminator_loss_all),np.array(generate_loss)),end=' ')
                    tf.print('%.2f s/step'%(duration))
                
                #-----------------------------------------
                #       plot the train history           |
                #-----------------------------------------
                #\\
                if step % 5 == 0 :
                    history.append([step, discriminator_loss_all, generate_loss])
                    
                #-----------------------------------------
                #       save the model_weights           |
                #-----------------------------------------
                #\\
                if step % self.traindata.save_step == 0 and step != 0:
                    # save the train steps
                    np.save(os.path.join(self.traindata.model_path,'./history.npy'), history)
                    self.combineModel.save(os.path.join(self.traindata.model_path,'gan.h5'))
                    
                #-----------------------------------------
                #       save the image of generate       |
                #-----------------------------------------
                #\\
                if step % 50 == 0 and step != 0:
                    noise_test = np.random.normal(size=[1,self.traindata.noise_dim])
                    noise_test = np.array(noise_test,np.float32)
                    fake_image = self.generate_model(noise_test)
                    '''
                        复原图像
                        1.乘以255后需要映射成uint8的类型
                        2.也可以保持[0,1]的float32类型，依然可以直接输出
                    '''
                    arr_img = np.array([fake_image],np.float32).reshape([self.traindata.dim,self.traindata.dim,3])*255
                    arr_img = np.array(arr_img,np.uint8)
                    
                    #保存为tfrecords用的是PIL.Image,即打开为RGB，所以在用cv显示时需要转换为BGR
                    arr_img = cv.cvtColor(arr_img,cv.COLOR_RGB2BGR)
                    cv.imwrite(self.traindata.save_path+str(step)+'.jpg',arr_img)

            except tf.errors.OutOfRangeError: 
                tf.print("the iter is out of range\n")
    

if __name__=='__main__':
    traindata = datasets(args.data,args.type)
    mygan = wgan(traindata)
    mygan.build()
    mygan.train()
