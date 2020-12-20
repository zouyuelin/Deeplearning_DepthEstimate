#! -*- coding: utf-8 -*-
'''
    Designer: zyl
    use :
    python gan.py --data [image path] --type ['gp' or 'div']
'''
import time
import numpy as np
import tensorflow as tf
import keras  
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import sys

noise_dim = 128
dim = 64
epochs = 1000
batch_size = 64
data_num = 12500
learning_rate = 2e-4
save_step = 300
n_critic = 1
n_generate = 1
tfrecords_path = 'data/train.tfrecords'
save_path = 'image/'
model_path = 'model/'
#log_path = 'log/'


tf.app.flags.DEFINE_string(
    'data', 'None', 'where the datas?.')
tf.app.flags.DEFINE_string(
    'type', 'gp', 'what is the type?.')
FLAGS = tf.app.flags.FLAGS

if(FLAGS.data == None):
    os._exit(0)
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('image'):
    os.mkdir('image')
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('model'):
    os.mkdir('model')
#if not os.path.exists('log'):
#   os.mkdir('log')

#-------------------------------------------------------------------
#                        create the tfrecords                      |
#-------------------------------------------------------------------  

def create_tfrecords():
    if os.path.exists(tfrecords_path):
        return 0
    if(FLAGS.data == None):
        print('the data is none,use: python gan.py --data []')
        os._exit(0)
    writer_train= tf.python_io.TFRecordWriter(tfrecords_path)
    value = 0
    object_path = FLAGS.data
    total = os.listdir(object_path)
    num = len(total)
    num_i = 1
    value = 0
    print('-----------------------------making dataset tfrecord,waiting--------------------------')
    for index in total:
        img_path=os.path.join(object_path,index)
        img=Image.open(img_path)
        img=img.resize((dim,dim))
        img_raw=img.tobytes()
        
        '''
            it is on my datasets, please delete 74-75 codes! 
        '''
        
        if 'cat' in index:
            value = 0
        else:
            continue
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
        writer_train.write(example.SerializeToString())  #序列化为字符串
        sys.stdout.write('--------%.4f%%-----'%(num_i/float(num)))
        sys.stdout.write('%\r')
        sys.stdout.flush()
        num_i = num_i +1
    print('-------------------------------datasets has completed-----------------------------------')
    writer_train.close()

    
#-------------------------------------------------------------------
#                            datatfrecords                         |
#-------------------------------------------------------------------   
def load_image(serialized_example):   
    features={
        'label': tf.io.FixedLenFeature([], tf.int64),
        'img_raw' : tf.io.FixedLenFeature([], tf.string)}
    parsed_example = tf.io.parse_example(serialized_example,features)
    image = tf.decode_raw(parsed_example['img_raw'],tf.uint8)
    image = tf.reshape(image,[-1,dim,dim,3])
    image = tf.cast(image,tf.float32)*(1./255)
    label = tf.cast(parsed_example['label'], tf.int32)
    label = tf.reshape(label,[-1,1])
    return image,label
 
def dataset_tfrecords(tfrecords_path,use_keras_fit=True): 
    #是否使用tf.keras fit函数
    if use_keras_fit:
        epochs_data = 1
    else:
        epochs_data = epochs
    dataset = tf.data.TFRecordDataset([tfrecords_path])
    '''
        这个可以有多个组成[tfrecords_name1,tfrecords_name2,...],可以用os.listdir(tfrecords_path):
    '''
    dataset = dataset\
                .repeat(epochs_data)\
                .shuffle(1000)\
                .batch(batch_size)\
                .map(load_image,num_parallel_calls = 8)
    #注意一定要将shuffle放在batch前      

    iter = dataset.make_initializable_iterator()#make_one_shot_iterator()
    train_datas = iter.get_next() #用train_datas[0],[1]的方式得到值
    return train_datas,iter
 

#-------------------------------------------------------------------
#                            define resBlock                       |
#-------------------------------------------------------------------   
    
def convolutional2D(x,num_filters,kernel_size,resampling):
    if resampling is 'up':
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same',
                       kernel_initializer=keras.initializers.RandomNormal())(x)
        #x = keras.layers.Conv2DTranspose(num_filters,kernel_size=kernel_size, strides=2,  padding='same',
        #              kernel_initializer=keras.initializers.RandomNormal())(x)
    elif resampling is 'down':
        x = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2,  padding='same',
                       kernel_initializer=keras.initializers.RandomNormal())(x)

    return x
    
def ResBlock(x, num_filters, resampling):
    #F1,F2,F3 = num_filters
    X_shortcut = x

    #//BN_relu
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = keras.layers.Activation('relu')(x)
    
    #//up or down
    x = convolutional2D(x,num_filters,kernel_size=(3,3),resampling=resampling)
    
    #//BN_relu
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = keras.layers.Activation('relu')(x)
    
    #//cov2d
    x = keras.layers.Conv2D(num_filters, kernel_size=(3,3), strides=1,padding='same',
                       kernel_initializer=keras.initializers.RandomNormal())(x)
    #//add_shortcut
    X_shortcut = convolutional2D(X_shortcut,num_filters,kernel_size=(3,3),resampling=resampling)
    X_add = keras.layers.Add()([x,X_shortcut])

    return X_add


#-------------------------------------------------------------------
#                            define generator                      |
#-------------------------------------------------------------------   

def generate(resampling='up'):
    nosie = keras.layers.Input(shape=(noise_dim,))
    g = keras.layers.Dense(512*4*4)(nosie)
    g = keras.layers.Reshape((4,4,512))(g)
    #4*4*512
    g = ResBlock(g,num_filters=512,resampling=resampling)
    #8*8*512
    g = ResBlock(g,num_filters=256,resampling=resampling)
    #16*16*256
    g = ResBlock(g,num_filters=128,resampling=resampling)
    #32*32*128
    g = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(g)
    g = keras.layers.Activation('relu')(g)
    
    g = keras.layers.UpSampling2D()(g)
    g = keras.layers.Conv2D(3, kernel_size=(3,3), strides=1, padding='same',
                       kernel_initializer=keras.initializers.RandomNormal())(g)
    #64*64*64
    g_out = keras.layers.Activation('tanh')(g)
    g_model = keras.Model(nosie,g_out)
    return g_model

#-------------------------------------------------------------------
#                            define discriminator                  |
#-------------------------------------------------------------------  

def discriminator(resampling='down'):
    real_in = keras.layers.Input(shape=(dim, dim, 3))
    #d = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same',strides=1,
    #                  kernel_initializer=keras.initializers.RandomNormal())(real_in)
    #64*64*64
    d = ResBlock(real_in,num_filters=128,resampling=resampling)         
    #32*32*128
    d = ResBlock(d,num_filters=256,resampling=resampling)
    #16*16*256
    d = ResBlock(d,num_filters=512,resampling=resampling)
    #8*8*512
    d = keras.layers.Conv2D(512, kernel_size=(3,3), padding='same',strides=2,
                       kernel_initializer=keras.initializers.RandomNormal())(d)
    d = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(d)
    d = keras.layers.Activation('relu')(d)

    #4*4*512
    '''
        GlobalAveragePooling :it can replace the full connection layer
        you can use the Dense to test the network
    '''
    d = keras.layers.GlobalAveragePooling2D()(d)
    d_out = keras.layers.Dense(1,use_bias = False)(d)
    d_model = keras.Model(real_in,d_out)
    return d_model

#-------------------------------------------------------------------
#                           show process of trian                  |
#------------------------------------------------------------------- 
def plot(history):
    history = np.array(history)
    plt.ion()
    plt.figure(figsize=(12,4))
    plt.title('Train History')
    plt.plot(history[:,0],history[:,1])
    plt.ylabel('loss')
    plt.plot(history[:,0],history[:,2])
    plt.plot(history[:,0],history[:,3])
    plt.xlabel('step')
    plt.legend(['d_loss','distance','g_loss'],loc='upper left')
    plt.savefig(os.path.join(model_path,'history.png'))
    plt.pause(2)
    plt.close()
    
def main():
    #------------------------------
    #define the generate model    *
    #------------------------------
    generate_model = generate()
    
    #--------------------------------
    #define the discriminator model *
    #--------------------------------
    discriminator_model = discriminator()
    
    #cat the network
    discriminator_model.summary()
    generate_model.summary()
    
    #//
    #-------------------------------------------------------------------
    #                            train the Discriminator               |
    #-------------------------------------------------------------------
    #//
    '''
        you need to redefined the Input rather than use the Input previous
    '''
    #Input para
    Dx_real_img = keras.layers.Input(shape=(dim, dim, 3))
    Dz_noise = keras.layers.Input(shape=(noise_dim,))
    D_uniform = keras.layers.Input(shape=(1,1,1))
    
    #set the trainable 
    generate_model.trainable = False
    
    #get the score
    D_fake_img = generate_model(Dz_noise)
    D_fake_score = discriminator_model(D_fake_img)
    D_real_score = discriminator_model(Dx_real_img)
    
    #train net
    gan_train_d = keras.Model([Dx_real_img, Dz_noise, D_uniform],[D_real_score,D_fake_score])
    
    #set the loss function according to the algorithm
    k = 2
    p = 6
    u = D_uniform
    
    #then, get a new input consist from fake and real
    x_ = (1.-u)*Dx_real_img+u*D_fake_img
    
    #//
    #-------------------------------------------------------------------
    #                            wgan div loss function                |
    #                               n_critic = 1                       |
    #                          arxiv.org/pdf/1712.01026.pdf            |
    #-------------------------------------------------------------------
    #//
    if FLAGS.type == 'div':
        gradients = K.gradients(discriminator_model(x_), [x_])[0]
        grad_norm = K.sqrt(K.sum(gradients ** 2, axis=[1, 2, 3])) ** p
        grad_penalty = k * K.mean(grad_norm)
        discriminator_loss = K.mean(D_real_score - D_fake_score)
    
    
    #//
    #-------------------------------------------------------------------
    #                            wgan gp  loss function                |
    #                               n_critic = 5                       |
    #                          arxiv.org/pdf/1704.00028.pdf            |
    #-------------------------------------------------------------------
    #//
    if FLAGS.type == 'gp':
        gradients = K.gradients(discriminator_model(x_), [x_])[0]
        grad_norm = K.sqrt(K.sum(gradients ** 2, axis=[1, 2, 3]))
        grad_norm = K.square(1-grad_norm)
        grad_penalty =  10*K.mean(grad_norm)
        discriminator_loss = K.mean(D_fake_score-D_real_score)
    
    #//
    #tf.summary.scalar('discriminator_loss',discriminator_loss)
    
    #loss function
    discriminator_loss_all = grad_penalty+ discriminator_loss 
    #//
    #tf.summary.scalar('discriminator_loss_all',discriminator_loss_all)
    
    #compile the model
    gan_train_d.add_loss(discriminator_loss_all) #min
    gan_train_d.compile(optimizer=keras.optimizers.Adam(learning_rate, 0.5))
    gan_train_d.metrics_names.append('DistanceFromRealAndFake')
    gan_train_d.metrics_tensors.append(-discriminator_loss) #max
    
    #//
    #-------------------------------------------------------------------
    #                            train the Generator                   |
    #-------------------------------------------------------------------
    #//
    #Input para
    Gz_nosie = keras.layers.Input(shape=(noise_dim,))
    
    #set the trainable 
    discriminator_model.trainable = False
    generate_model.trainable = True
    
    #get the score
    G_fake_img = generate_model(Gz_nosie)
    G_fake_score = discriminator_model(G_fake_img)
    
    #train net
    gan_train_g = keras.Model(Gz_nosie,G_fake_score)
    
    #loss function

    if FLAGS.type == 'div':
        generate_loss = K.mean(G_fake_score)
    if FLAGS.type == 'gp':
        generate_loss = -K.mean(G_fake_score)#min this value
    #//
    #tf.summary.scalar('discriminator_loss',discriminator_loss)

    #compile the model
    gan_train_g.add_loss(generate_loss) #min
    gan_train_g.compile(optimizer=keras.optimizers.Adam(learning_rate, 0.5))
    
    #\\
    #---------------------------------------------------------------------
    #\\
    #cat the network
    gan_train_d.summary()
    gan_train_g.summary()
    
    #creat the session, get the dataset from tfrecords
    sess = tf.Session()
    train_datas,iter = dataset_tfrecords(tfrecords_path,use_keras_fit=False)
    sess.run(iter.initializer)
    
    #write to log
    #merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(log_path,sess.graph)
    
    print("-----------------------------------------start---------------------------------------")
    #continue
    if os.path.exists(os.path.join(model_path,'gan.weights')):
        gan_train_g.load_weights(os.path.join(model_path,'gan.weights'))
        if os.path.exists(os.path.join(model_path,'history.npy')):
            history = np.load(os.path.join(model_path,'./history.npy'), allow_pickle=True).tolist()
            #read the last data use -1 index,and use 0 to read the first data
            #\\
            last_iter = int(history[-1][0])
            print('Find the npy file, the last save iter:%d' % (last_iter))
        else:
            history = []
            last_iter = -1
    else:
        print('There is no .npy file, start a new file---------')
        history = []
        last_iter = -1
        
    #state the global vars
    #\\
    global n_critic
    global n_generate
    
    #the loop body
    #\\
    for step in range(last_iter+1,int(last_iter+1+epochs*data_num/batch_size+1)):
        try:
            #get the time
            start_time = time.time()
            
            #datasets
            train_datas_ = sess.run(train_datas)
            '''
                if the datasets' shape is not batch_size
            '''
            if train_datas_[0].shape[0] != batch_size:
                sess.run(iter.initializer)
                train_datas_ = sess.run(train_datas)
            
            z_noise = np.random.normal(size=batch_size*noise_dim)\
                                    .reshape([batch_size,noise_dim])
            u_niform = np.random.uniform(low=0.0,high=1.0,size=(batch_size,1,1,1))
            
            #-----------------------------------------
            #   phase 1 - training the discriminator |
            #-----------------------------------------
            #\\
            for step_critic in range(n_critic):
                d_loss,distance = gan_train_d.train_on_batch([train_datas_[0],z_noise,u_niform],None)
            
            #-----------------------------------------
            #   phase 2 - training the generator     |
            #-----------------------------------------
            #\\
            for step_generate in range(n_generate):
                g_loss = gan_train_g.train_on_batch(z_noise,None)
            
            
            #-----------------------------------------
            #        set the num of n_batch          |
            #-----------------------------------------
            #\\
            if g_loss >= 10.:
                n_generate = 5
                n_critic = 1
            elif g_loss <= -10.:
                n_generate = 1
                n_critic = 1
            
            #get loss log
            #summary = sess.run(merged)
            #summary_writer.add_summary(summary,step)
            
            #get the time 
            duration = time.time()-start_time
            
            #-----------------------------------------
            #            print the loss              |
            #-----------------------------------------
            if step % 5 == 0:
                print("The step is %s,d_loss:%s,distance:%s,g_loss:%s, n_generate:%s"%(step,d_loss,distance,g_loss,n_generate),end=' ')
                print('%.2f s/step'%(duration))
            
            #-----------------------------------------
            #       plot the train history           |
            #-----------------------------------------
            #\\
            if step % 5 == 0 :
                history.append([step, d_loss,distance, g_loss])
                  
            #-----------------------------------------
            #       save the model_weights           |
            #-----------------------------------------
            #\\
            if step % save_step == 0 and step != 0:
                # save the train steps
                np.save(os.path.join(model_path,'./history.npy'), history)
                gan_train_g.save_weights(os.path.join(model_path,'gan.weights'))
                plot(history)
                
            #-----------------------------------------
            #       save the image of generate       |
            #-----------------------------------------
            #\\
            if step % 50 == 0 and step != 0:
                noise_test = np.random.normal(size=[1,noise_dim])
                noise_test = np.cast[np.float32](noise_test)
                fake_image = generate_model.predict(noise_test,steps=1)
                '''
                    复原图像
                    1.乘以255后需要映射成uint8的类型
                    2.也可以保持[0,1]的float32类型，依然可以直接输出
                '''
                arr_img = np.array([fake_image],np.float32).reshape([dim,dim,3])*255
                arr_img = np.cast[np.uint8](arr_img)
                
                #保存为tfrecords用的是PIL.Image,即打开为RGB，所以在用cv显示时需要转换为BGR
                arr_img = cv2.cvtColor(arr_img,cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path+str(step)+'.jpg',arr_img)
                #cv2.imshow('fake image',arr_img)
                #cv2.waitKey(1500)#show the fake image 1.5s
                #cv2.destroyAllWindows()
        except tf.errors.OutOfRangeError: 
            sess.run(iter.initializer)
    plot(history)     
    #summary_writer.close()
    
create_tfrecords()
main()
    
