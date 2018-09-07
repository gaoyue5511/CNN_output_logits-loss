import os
import glob
from skimage import io,transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.cluster import KMeans

#以下是非监督聚类：Kmeans部分，主要用于图像预处理，和真实样本预处理。

def load_img(image_file):
#    image_file = cv2.imread(path)   # 修改了，不用读取，直接填入图片数据
    w,h,d = (image_file.shape)
    image = image_file.reshape(w*h , d)
    return w, h, d, image
    
def model(data, cluster_num, kmean_epoch):
    kmean = KMeans(n_clusters=cluster_num, random_state=0, max_iter=kmean_epoch).fit(data)
    labels = kmean.predict(data)
    return kmean, labels

def recreate_image(labels, w, h, d):
    new_image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            new_image[i][j] = labels[label_idx]
            label_idx += 1
    return new_image

#def show_img(result):
#    plt.imshow(result)
#    plt.show()


def kmean_poopoo(img_file, cluster_num, kmean_epoch):
    w, h, d, data = load_img(img_file)
    kmean, labels = model(data, cluster_num, kmean_epoch)
    result = recreate_image(labels, w, h, d)
    
    return result
#    show_img(result)
#    print(np.shape(result))
#    print(result)

# img = '13R.jpg'

cluster_num = 2
kmean_epoch = 100


# kmean_poopoo(img, cluster_num, kmean_epoch)

def read_img (path, data_size):
    imgs = []
    labels = []
    cate = []
    for x in os.listdir(path):            # 读取文件夹里所有文件夹的路径，赋值到cate列表
        if (os.path.isdir(path+'\\'+x)):
            cate.append(path+'\\'+x)
            
    # cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] 大神都这样写，但是不好理解，有些太perl
    
    for idx,folder in enumerate(cate):    #给文件夹排序号，0是0文件夹，1是1文件夹...
        for im in glob.glob(folder+'/*.jpg'): #遍历文件夹内的*.jpg文件（路径）
#            print(im)
            img = cv2.imread(im)  #读取jpg文件
#            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #把图片转换为黑白(彩色是3维，黑白是2维)
            img = cv2.resize(img,(data_size[0],data_size[1]),interpolation=cv2.INTER_CUBIC)
            img = kmean_poopoo(img, cluster_num, kmean_epoch)
            print(np.shape(img))
            # 根据要求调整图片大小
#            img = 255 - img                 # 以下为图片的处理 ，先减去一个255
#            for row in range(img.shape[0]):
#                for col in range(img.shape[1]):
#                    if img[row][col] < 120.0:   #把灰度120以下的化为0(降噪)
#                        img[row][col] = 0
                        
#            img = img / 255  # 把各个直降到1-0之间的小数
            imgs.append(img) # 追加到数组imgs
            labels.append(idx) # 追加到数组labels
#            plt.imshow(img,'gray')
#            plt.show()
#            print(np.shape(labels))
#            print(np.shape(img))  #注意img是单张图片是32*32的二维数组 
#            print(np.shape(imgs)) #但是imgs是：第几张图片*32*32 的三维数组,现在我有54张图片，所以是 54*32*32
#            print (labels)         #标签的顺序也是依照图片的顺序匹配的，也就是：imgs[0] 这个图片的标签是labels[0]
#            print (len(labels))    #相对的labels的总数，到现在为止是160个标签
#            因为在给卷积函数数据时是[每批的样本数量，样本长，样本宽，样本通道数]，如果是灰度图，是没有通道数的，需要增加一维，此处我想做彩色的，所以把之前的彩色转换为灰度，注释掉了。
    np.asarray(imgs,np.float32),np.asarray(labels,np.float32)  # 将数组转化为矩阵
    return np.asarray(imgs,np.float32),np.asarray(labels,np.float32)




def data_split_flow (data,label,ratio):# data是读取的图片集合，label是图片的标签集合，ratio是你想要(百分之)多少数据用于培训.
    num_example = data.shape[0]  # 这个data是读取图片的合计，其中第一维就是图片的序号，也就是图片的总量
    arr = np.arange(num_example) # np.arange(起始值，终止值，步长) 与arange(起始值，终止值，步长) 不同之处是np.arange的参数可以是小数，这里应该是np.arange(28)
    np.random.shuffle(arr) #随机打乱顺序函数，多维矩阵中，只对第一维做打乱顺序操作。也就是np.arange(28)中的顺序被随机打乱
#    print (type(arr))
#    print (type(data))
#    print (data.shape)
    data = data[arr]  # 因为arr现在是一维的随机化的np矩阵，用它可以覆盖掉原数据的第一维，也就是重新给data排序
    label = label[arr] # 同理，也同样使label标签随机化，这两次随机化的参数arr是相同的，也就是随机后的数据和标签是可以对上号的。
#    print (data.shape)
    s = np.int(num_example*ratio)  # 图片总数*想要取走百分之多少，并且取整，然后赋予s
    
    x_train = data[:s]  #以下是把图片分为“训练用图片”“训练用图片的标签”，“验证用图片”“验证用图片的标签”。其中[:s]或[s:]是列表的切片，表示由开始到s，或由s到最后。
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    
    return x_train,y_train,x_val,y_val
#c,d,e,f = data_split_flow(a,b,0.8)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False): #四个参数是：训练数据，测试数据，用户输入的每批训练的数据数量，shuffle是洗牌的意思，这里表示是否开始随机。
    assert len(inputs) == len(targets)  #assert断言机制，如果后面的表达式为真，则直接抛出异常。在这里的意思,大概就是:样本和标签数量要对上
    if shuffle:
        indices = np.arange(len(inputs)) #生成一个np.arange可迭代长度是len(训练数据),也就是训练数据第一维数据的数量(就是训练数据的数量，训练图片的数量)。
        np.random.shuffle(indices)  #np.random.shuffle打乱arange中的顺序，使其随机循序化，如果是数组，只打乱第一维。
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size): # 这个range(初始值为0，终止值为[训练图片数-每批训练图片数+1]，步长是[每批训练图片数])：例(0[起始值],80[训练图片数]-20[每批训练图片数],20[每批训练图片数]),也就是(0,60,20)当循环到60时,会加20到达80的训练样本.
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size] # 如果shuffle为真,将indices列表,切片(一批)赋值给excerpt
        else:
            excerpt = slice(start_idx, start_idx + batch_size) # 如果shuffle为假,将slice()函数(切片函数),实例化,初始值为start_idx,结束为止为start_idx + batch_size(也就是根据上一批起始,算出本批结束的位置.),间距为默认.
        yield inputs[excerpt], targets[excerpt]
        #yield常见用法：该关键字用于函数中会把函数包装为generator。然后可以对该generator进行迭代: for x in fun(param).
        #按照我的理解，可以把yield的功效理解为暂停和播放。
        #在一个函数中，程序执行到yield语句的时候，程序暂停，返回yield后面表达式的值，在下一次调用的时候，从yield语句暂停的地方继续执行，如此循环，直到函数执行完。
        #此处,就是返回每次循环中 从inputs和targets列表中,截取的 经过上面slice()切片函数定义过的 数据.
        #(最后的shuffle变量，决定了样本是否随机化)

def cnn_fc (input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'): # 开启一个联系上下文的命名空间，空间名是layer1-conv1，在tf.get_variable可以顺利调用
        conv1_weights = tf.get_variable('weight',[5,5,3,6],initializer = tf.truncated_normal_initializer(stddev = 0.1))
        #上面一行命令是生成卷积核：是一个tansor类型，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
            # tf.truncated_normal_initializer：从截断的正态分布中输出随机值。这是神经网络权重和过滤器的推荐初始值。
            # mean：一个python标量或一个标量张量。要生成的随机值的均值。
            # stddev：一个python标量或一个标量张量。要生成的随机值的标准偏差。
            # seed：一个Python整数。用于创建随机种子。查看 tf.set_random_seed 行为。
            # dtype：数据类型。只支持浮点类型。
        conv1_biases = tf.get_variable("bias",[6],initializer = tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        # 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
        #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
        #具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
        
        #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        #具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
        
        #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
        #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
        #第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true#    
        #结果返回一个Tensor，这个输出，就是我们常说的feature map特征图，shape仍然是[batch, height, width, channels]这种形式。
        
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        # 激活函数，非最大值置零
        # 这个函数的作用是计算激活函数 relu，即 max(features, 0)。即将矩阵中每行的非最大值置0。
        
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1,ksize = [1,2,2,1], strides=[1,2,2,1],padding="VALID")
        #tf.nn.max_pool(value, ksize, strides, padding, name=None)
        #参数是四个，和卷积很类似：
        #第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
        #第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        #第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        #第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
        #返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
        
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,6,16],initializer=tf.truncated_normal_initializer(stddev=0.1))# [5,5,32,64] 5表示本次卷积核高宽，32表示经过上一层32个卷积核的卷积，我们有了32张特征图，64表明本次会有64个卷积核卷积
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 8*8*16
        reshaped = tf.reshape(pool2,[-1,nodes])  # 其中pool2是链接上一层，我把pool4和pool3的卷积核池化层删除了，卷的太多都要成渣渣了。
        # tf.reshape(tensor(矩阵),shape(维度),name=None)
        # 改变一个矩阵的维度，可以从多维变到一维，也可以从一维变到多维
        # 其中，-1参数表示不确定，可由函数自己计算出来，原矩阵/一个维度=另一个维度
        
    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        #在深度学习中，通常用这几个函数存放不同层中的权值和偏置参数，
        #也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)得到norm后，都放到’regular’的列表中作为正则项，
        #然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss。
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases) # MCP模型
        if train: fc1 = tf.nn.dropout(fc1, 0.5) #tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
        
        # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None) 
        # 上面方法中常用的是前两个参数：
        # 第一个参数x：指输入的数据。
        # 第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符,  keep_prob = tf.placeholder(tf.float32) 。
        # tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
        # 第五个参数name：指定该操作的名字。


    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 10],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
#        variable_summaries(fc3_weights)
#        variable_summaries(fc3_biases)

    return logit





# 设置超参
path = "E:\\study\\MINST-PLUS\\data" #样本路径
data_size = [32,32,3]                #样本大小和通道数
data_ratio = 0.8   # 百分之多少用于训练，剩下的用于测试
is_train = False
epoch = 1           # 训练次数
batch_size = 1      # 每次训练多少



# 样本和标签的读入与分类
data,label = read_img(path,data_size)
x_train,y_train,x_val,y_val = data_split_flow(data,label,data_ratio)

# 为数据与标签设立两个存放空间
x = tf.placeholder(tf.float32,shape=[None,data_size[0],data_size[1],data_size[2]],name = 'x')
y_ = tf.placeholder(tf.int32,shape=[None,],name = 'y_')

# 定义规则化方法，并计算网络激活值
regularizer = tf.contrib.layers.l2_regularizer(0.0001)   # 过拟合与正则化(regularizer)，这个regularizer就是inference函数的最后一个参数。
#两种思想都是希望限制权重的大小，使得模型不能拟合训练数据中的随机噪点。(两种思想，就是两个公式，因为是图，就没贴出来)
#两种方式在TensorFlow中的提供的函数为：
#tf.contrib.layers.l1_regularizer(scale, scope=None) 其中scale为权值(这个权值会乘以w的值，MCP的内个w，江湖传闻w和过拟合值有说不清的关系)
#tf.contrib.layers.l2_regularizer(scale, scope=None)

logits = cnn_fc(x,is_train,regularizer)  #x是输入的图像的tansor，logits是经过卷积、池化、全连接处理处理过的数据

#b = tf.constant(value=1,dtype=tf.float32)  # constant（值、列表 ， 数组格式）根据值、列表，生成一个数组，格式为“数组格式”
#logits_eval = tf.multiply(logits,b,name='logits_eval') # 额，不知道这是计算啥


# 计算误差与准确率，并写入日志 （我没有日志，呵呵）
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits) #计算logits 和 labels 之间的稀疏softmax 交叉熵 这个是计算误差率
'''
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)  # tf.train.AdamOptimizer 优化器中的梯度优化函数，
# 作用是依据learning_rate步长，来最小化loss误差率。

correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
#tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，
#这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

# tf.cast(x, dtype, name=None)
# 此函数是类型转换函数
# 参数
# x：输入
# dtype：转换目标类型
# name：名称
# 返回：Tensor

# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，
# 返回的值的矩阵维度和A是一样的，返回的也是一个矩阵、向量、列表，里面都是true和false。

#这一行的意思，大概是，通过以上三个函数，对比处理后的logits值和labels值，然后得出一个判断表单

acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #这个......大概是在计算准确率
# 求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 参数1--input_tensor:待求值的tensor。
# 参数2--reduction_indices:在哪一维上求解。
# 参数（3）（4）可忽略

# tf.summary.scalar('accuracy', acc)

'''

# 创建保存点，并进入计算图流程   还有限制gpu，我的电脑没有这句话就各种死

#saver=tf.train.Saver()
sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))  #限制gpu内存
sess.run(tf.global_variables_initializer())   # 全世界get_variable嗨起来

for epoch in range(epoch):   # 训练多少遍，epoch是用户输入的，比如是5，也就是把样本遍历5遍
    start_time = time.time()       # 开始计时
    
    ### 单次训练部分  此处for循环结束之日，就是训练样本遍历了一遍之时......
#    train_loss, train_acc, n_batch = 0, 0, 0    # 先定义下训练误差，训练识别率，训练批次
    for x_train_a, y_train_a in minibatches(x_train, y_train,batch_size, shuffle=False):  
    #遍历minibatches函数，因为这个函数中有yield关键字，每次for，会获取到不同的批次，直到训练样本集合身体被掏空。注意，这里shuffle为True，样本是随机的。
        log,los = sess.run([logits,loss], feed_dict={x: x_train_a, y_: y_train_a})  #向sess.run中喂数据，
        # 其中merged是train_summary计算图；train_op是梯度优化方法，err接收的loss是误差率；ac接收的acc是准确率。后面的x和y_就是每批的数据和标签。
#        train_loss += err; train_acc += ac; n_batch += 1  #统计误差率、准确率、批次
        print(log)
        print(y_train_a)
        print(los)  #输出每个样本的logits值和loss值
    print()
    print()



sess.close()