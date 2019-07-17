import tensorflow as tf 

class Model:
    # model constants
    batchSize = 64
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, restore=False):
        " initialize model: add CNN, RNN and CTC layers"
        self.charList = charList
        self.restore = restore
        self.ID = 0
        # use normalization over a minibatch
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        # input images
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
        # setup CNN, RNN and CTC layers
        self.CNN_layer()
        self.RNN_layer()
        self.CTC_layer()
        # setup optimizer to train neural network 
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=())
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        self.sess, self.saver = self.setup()
    
    def CNN_layer(self):
        ''' create CNN layers and reuturn output of these layers '''
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
        # list of parameters for the layers
        kernelSizes = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        pool = cnnIn4d # input for first CNN layer
        for i in range(numLayers):
            name = 'conv_' + str(i+1)
            # input: shape = (?, 128, 32, 1)
            # conv_1: shape = (?, 128, 32, 32)
            # conv_2: shape = (?, 64, 16, 64)
            # conv_3: shape = (?, 32, 8, 128)
            # conv_4: shape = (?, 32, 4, 128)
            # conv_5: shape = (?, 32, 2, 256)
            with tf.variable_scope(name):
                weights_shape = [kernelSizes[i], kernelSizes[i], featureVals[i], featureVals[i+1]]
                weights = tf.get_variable(name='_weights', shape=weights_shape)
                biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[featureVals[i+1]]))
                conv = tf.nn.conv2d(input=pool, filter=weights, padding='SAME', strides=(1, 1, 1, 1))
                conv = tf.nn.bias_add(conv, biases, name='pre-activation')
                conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
                relu = tf.nn.relu(conv_norm, name='activation')
            # pool_1: shape = (?, 64, 16, 32)
            # pool_2: shape = (?, 32, 8, 64)
            # pool_3: shape = (?, 32, 4, 128)
            # pool_4: shape = (?, 32, 2, 128)
            # pool_5: shape = (?, 32, 1, 256)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
        self.cnnOut4d = pool # shape = (?, 32, 1, 256)
    
    def RNN_layer(self):
        ''' create RNN layers and return ouput of these layers '''
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2]) # shape = (?, 32, 256)
        numHidden = 256
        self.keepprob = tf.placeholder(tf.float32, name='keepprob')
        # basic cells which is used to build RNN
        cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True), output_keep_prob=self.keepprob) for _ in range(2)] # 2 layers
        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        # build independent forward and backward bidirectional RNNs
        # return two output sequences fw and bw, shape = (?, 32, 256)
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
        # concatenate to form a sequence shape = (?, 32, 1, 512)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        # project output to chars (including blank), return shape = (?, 32, len(self.charList)+1)
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
    
    def CTC_layer(self):
        ''' create CTC loss and decoder '''
        self.ctcIn3d = tf.transpose(self.rnnOut3d, [1, 0, 2]) # shape = (32, ?, len(self.charList)+1)
        # ground truth texts as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
        # calculate loss for minibatch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3d, sequence_length=self.seqLen, ctc_merge_repeated=True))
        # best path decoder
        self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3d, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
    
    def setup(self):
        ''' initialize TensorFlow '''
        sess = tf.Session()
        # saver saves model to file
        saver = tf.train.Saver(max_to_keep=1)
        modelFolder = '../model/'
        latestSaved = tf.train.latest_checkpoint(modelFolder)

        if self.restore and latestSaved:
            # load saved model if available
            saver.restore(sess, latestSaved)
        else:
            sess.run(tf.global_variables_initializer())
        return sess, saver
    
    def toSparse(self, texts):
        ''' put ground truth texts into sparse tensor for ctc_loss '''
        indices, values = ([] for _ in range(2))
        shape = [len(texts), 0]

        for batchElement, text in enumerate(texts):
            # convert to string of label
            labelStr = [self.charList.index(c) for c in text]
            if len(labelStr) > shape[1]:
                # sparse tensor must have size of maximum
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for i, label in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)
        
        return indices, values, shape
    
    def decoderOutputToText(self, ctcOutput, batchSize):
        ''' extract texts from output of CTC decoder '''
        # string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]
        # ctc returns tuple, first element is sparse tensor
        decoded=ctcOutput[0][0]
        # go over all indices and save mapping
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]
            encodedLabelStrs[batchElement].append(label)
        # map labels to chars
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]
    

    def trainBatch(self, minibatch):
        ''' training neural network '''
        numBatchElements = len(minibatch.imgs)
        sparse = self.toSparse(minibatch.gtTexts)
        # decay learning rate from 0.01 to 0.0001
        lr = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
        feed = {self.inputImgs : minibatch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : lr, self.keepprob: 0.75, self.is_train: True}
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed)

        self.batchesTrained += 1
        return loss
    
    def inferBatch(self, minibatch):
        ''' recognize the texts '''
        numBatchElements = len(minibatch.imgs)
        feed = {self.inputImgs : minibatch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.keepprob: 1.0, self.is_train: False}
        decoded = self.sess.run(self.decoder, feed_dict=feed)
        
        texts = self.decoderOutputToText(decoded, numBatchElements)
        return texts
    
    def save(self):
        ''' save model to file '''
        self.ID += 1
        self.saver.save(self.sess, '../model/crnn-model', global_step=self.ID)