[mxnet](https://github.com/dmlc/mxnet)一键式集成开发平台
-----

本工程是[mxnet训练自己的数据集](https://github.com/imistyrain/mxnet-mr)的升级版，重新梳理了代码结构，修正了之前存在的Bug，并且代码库更新到最新版。


1.mxnet安装，参照[官网](http://mxnet.readthedocs.org/en/latest/how_to/build.html)和[mxnet VS2013编译V2.0.pdf](mxnet VS2013编译V2.0.pdf)。

2.数据准备

数据是参考ImageNet的格式组织的，也就是所有的图片按照类别分组，相同类别的放入同一文件夹下，最后这些不同类别的文件夹放到一个文件夹（示例为platechars，是开源项目[EasyPR](https://github.com/liuruoze/EasyPR)中车牌字符的训练样本）。

接下来需要转换为mxnet所需的格式，也就是rec格式，mxnet自带的im2rec.py就是完成此功能的，此外，还需要生成训练样本的字典文件synset.txt，也就是有哪些类别。

完成以上这些功能的代码被封装成了preprocess.bat：

```python
	echo "Generating lst"
	python preprocess/im2rec.py platechars/chars platechars --recursive=True --list=True
	echo "Generating rec"
	python preprocess/im2rec.py platechars/chars platechars
	echo "Generating synsetwords"
	python preprocess/generatesynsetwords.py --datadir=platechars
```

3.定义模型
需要定义网络的结构，训练时batch大小、迭代次数和学习率等和训练相关的参数。
示例采用的是lenet的结构，
```
def get_lenet():
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=34)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
```

4.训练模型

定义训练相关的参数，例如batch大小、迭代次数以及学习率和使用的GPU等。

```
def main():
    batch_size=1000
    num_epoch = 200
    num_gpus = 1
    logging.basicConfig(level=logging.DEBUG)
    gpus = [mx.gpu(i) for i in range(num_gpus)]
    lenet=get_lenet()
    model = mx.model.FeedForward(ctx=gpus, symbol=lenet, num_epoch=num_epoch,
                                     learning_rate=0.01, momentum=0.9, wd=0.0001,
                                     initializer=mx.init.Uniform(0.07))
    train_dataiter = mx.io.ImageRecordIter(
            path_imgrec=datadir+"/chars_train.rec",
            mean_img=datadir+"/mean.bin",
            rand_crop=True,
            rand_mirror=True,
            data_shape=(3,20,20),
            batch_size=batch_size,
            preprocess_threads=1)
    test_dataiter = mx.io.ImageRecordIter(
            path_imgrec=datadir+"/chars_val.rec",
            mean_img=datadir+"/mean.bin",
            rand_crop=False,
            rand_mirror=False,
            data_shape=(3,20,20),
            batch_size=batch_size,
            preprocess_threads=1)
    model.fit(X=train_dataiter, eval_data=test_dataiter,
              batch_end_callback=mx.callback.Speedometer(100))
    model.save(datadir+'/lenetweights',num_epoch)
```

 前述两部分的代码可通过train.bat进行训练，生成的模型文件存于platachars/lenetweights-0200.params，其中200是迭代的次数。

5.评估模型

实际应用中需根据训练的结果对训练样本的分类结果进行调整以提升分类精度，因此找到错误分类的样本就显得尤为重要。

evaluation.bat会读取训练样本所在的文件夹，遍历输出分类结果，并把所有分错的样本保存在error下相应的文件夹下，每个错误样本的命名由三部分组成：
原始文件名 + "\_" + 真值 + "\_" + 预 测值。

例如evaluation/error下1的文件夹中，397_0_1.jpg代表其原始文件为1下的397.jpg，本来应该是0，训练模型错误的将其分类为1。


-----

以上所有步骤均可自动化实现，因此封装了oneclick.bat，可一键式预处理、训练和评估。