import tensorflow as tf
import numpy as np


class TrafficSignModel(tf.keras.Model):
    """Model class representing our Traffic Sign Model."""

    def __init__(self):
        super(Model, self).__init__()

        # Initialize hyperparameters and Optimizer
        self.base_learning_rate = 0.01
        self.decay_steps = 100000
        self.decay_rate = 0.1
        self.momentum = 0.9
        self.weight_regularizer = 0.0005
        self.batch_size = 100
        self.epochs = 1450000

        self.reg = tf.keras.regularizers.L2(l2=self.weight_regularizer)

        self.optimizer_bias_2_0 = tf.keras.optimizers.SGD(learning_rate = 2*self.base_learning_rate, momentum = self.momentum)

        self.optimizer_bias_20_0 = tf.keras.optimizers.SGD(learning_rate = 20*self.base_learning_rate, momentum = self.momentum)

        self.learning_rate_conv_1_1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = self.base_learning_rate, decay_steps = self.decay_steps, decay_rate = self.decay_rate)
        self.optimizer_conv_1_1 = tf.keras.optimizers.SGD(learning_rate = self.learning_rate_conv_1_1, momentum = self.momentum)

        self.learning_rate_conv_5_01 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 5*self.base_learning_rate, decay_steps = self.decay_steps, decay_rate = 0.1*self.decay_rate)
        self.optimizer_conv_5_01 = tf.keras.optimizers.SGD(learning_rate = self.learning_rate_conv_5_01, momentum = self.momentum)

        self.learning_rate_conv_20_01 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 20*self.base_learning_rate, decay_steps = self.decay_steps, decay_rate = 0.1*self.decay_rate)
        self.optimizer_conv_20_01 = tf.keras.optimizers.SGD(learning_rate = self.learning_rate_conv_20_01, momentum = self.momentum)

        # Initialize trainable parameters
        self.conv1 = tf.keras.layers.Conv2D(filters = 96, kernel_size = 11, strides = (4,4),padding = 'VALID', activation = 'relu',name = 'conv1', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Ones(),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = 5, strides = (1,1),padding = 'VALID', activation = 'relu',name = 'conv2', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Ones(),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters = 384, kernel_size = 3, strides = (1,1),padding = 'VALID', activation = 'relu',name = 'conv3', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0.1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.conv4 = tf.keras.layers.Conv2D(filters = 384, kernel_size = 3, strides = (1,1),padding = 'VALID', activation = 'relu',name = 'conv4', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0.1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.conv5 = tf.keras.layers.Conv2D(filters = 384, kernel_size = 3, strides = (1,1),padding = 'VALID', activation = 'relu',name = 'conv5', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0.1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)
        self.conv6 = tf.keras.layers.Conv2D(filters = 4096, kernel_size = 6, strides = (1,1),padding = 'VALID', activation = 'relu',name = 'conv6', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.dropout1 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv7 = tf.keras.layers.Conv2D(filters = 4096, kernel_size = 1, strides = (1,1), padding = 'VALID', activation = 'relu',name = 'conv7', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.dropout2 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv8 = tf.keras.layers.Conv2D(filters = 256, kernel_size = 1, strides = (1,1), padding = 'VALID', name = 'conv8', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.conv9 = tf.keras.layers.Conv2D(filters = 4096, kernel_size = 1, strides = (1,1), padding = 'VALID', activation = 'relu',name = 'conv9', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.dropout3 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv10 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 1, strides = (1,1), padding = 'VALID', name = 'conv10', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.conv11 = tf.keras.layers.Conv2D(filters = 4096, kernel_size = 1, strides = (1,1), padding = 'VALID', activation = 'relu',name = 'conv11', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=1),kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.dropout4 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv12 = tf.keras.layers.Conv2D(filters = 1004, kernel_size = 1, strides = (1,1), padding = 'VALID', name = 'conv12', use_bias = True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=tf.keras.initializers.Constant(value=0),kernel_regularizer=self.reg, bias_regularizer=self.reg)
     

    def call(self, inputs):
        """
        Computes the forward pass of your network.
        """

        layer1 = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,0]])
        layer1 = self.conv1(layer1)
        layer1 = tf.nn.local_response_normalization(layer1, depth_radius=5, bias=2, alpha=0.0005, beta=0.75)
        layer1 = self.pool1(layer1)
        layer2 = tf.pad(layer1, [[2,2],[2,2],[2,2],[2,2]])
        layer2 = self.conv2(layer2)
        layer2 = tf.nn.local_response_normalization(layer1, depth_radius=5, bias=8, alpha=0.0005, beta=0.75)
        layer2 = self.pool2(layer2)
        layer3 = tf.pad(layer2, [[1,1],[1,1],[1,1],[1,1]])
        layer3 = self.conv3(layer3)
        layer4 = tf.pad(layer3, [[1,1],[1,1],[1,1],[1,1]])
        layer4 = self.conv4(layer4)
        layer5 = tf.pad(layer4, [[1,1],[1,1],[1,1],[1,1]])
        layer5 = self.conv5(layer5)
        layer5 = self.pool3(layer5)
        layer6 = tf.pad(layer5, [[3,3],[3,3],[3,3],[3,3]])
        layer6 = self.conv6(layer6)
        layer6 = self.dropout1(layer6)

        #Branch 1:
        branch1_layer7 = tf.pad(layer6, [[0,0],[0,0],[0,0],[0,0]])
        branch1_layer7 = self.conv7(branch1_layer7)
        branch1_dropout2 = self.dropout2(branch1_layer7)
        branch1_layer8 = tf.pad(branch1_dropout2, [[0,0],[0,0],[0,0],[0,0]])
        branch1_layer8 = self.conv8(branch1_layer8)
        batch_size =branch1_layer8,shape[0]
        reshape_height = branch1_layer8,shape[1] * 8
        reshape_weight = branch1_layer8,shape[2] * 8
        reshape_channels = branch1_layer8,shape[3]/64
        output_bb = tf.reshape(branch1_layer8, [batch_size,reshape_height,reshape_weight,reshape_channels]) 

        #Branch 2: Pixel
        branch2_layer9 = tf.pad(layer6, [[0,0],[0,0],[0,0],[0,0]])
        branch2_layer9 = self.conv9(branch2_layer9)
        branch2_dropout3 = self.dropout3(branch2_layer9)
        branch2_layer10 = tf.pad(branch2_dropout3, [[0,0],[0,0],[0,0],[0,0]])
        branch2_layer10 = self.conv10(branch2_layer10)
        batch_size =branch2_layer10,shape[0]
        reshape_height = branch2_layer10,shape[1] * 8
        reshape_weight = branch2_layer10,shape[2] * 8
        reshape_channels = branch2_layer10,shape[3]/64
        output_pixel = tf.reshape(branch2_layer10, [batch_size,reshape_height,reshape_weight,reshape_channels])

        #Branch 3:
        branch3_layer11 = tf.pad(layer6, [[0,0],[0,0],[0,0],[0,0]])
        branch3_layer11 = self.conv11(branch3_layer11)
        branch3_dropout4 = self.dropout4(branch3_layer11)
        branch3_layer12 = tf.pad(branch3_dropout4, [[0,0],[0,0],[0,0],[0,0]])
        branch3_layer12 = self.conv12(branch3_layer12)
        batch_size =  branch3_layer12.shape[0]
        reshape_height = branch3_layer12,shape[1] * 2
        reshape_weight = branch3_layer12,shape[2] * 2
        reshape_channels = branch3_layer12,shape[3]/4
        output_type = tf.reshape(branch3_layer12, [batch_size,reshape_height,reshape_weight,reshape_channels])

        return output_bb, output_pixel, output_type

    def loss(self, output_bb, output_pixel, output_type, label_bb, label_pixel, label_type):
        
        # TODO: Bounding Box Loss

        loss_pixel_calculator = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_pixel = loss_pixel_calculator(label_pixel, tf.nn.softmax(output_pixel))

        loss_type_calculator = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_type = loss_type_calculator(label_type, tf.nn.softmax(output_type))

        return 1*loss_pixel + 3*loss_type

    def accuracy_function(self, output_bb, output_pixel, output_type, label_bb, label_pixel, label_type):
        """
        Computes the accuracy across a batch of logits and labels.
        """
        # TODO: Bounding Box Accuracy

        acc_pixel = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_pixel, 1), tf.argmax(label_pixel, 1)), tf.float32))

        acc_type = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_type, 1), tf.argmax(label_type, 1)), tf.float32))

        return acc_bb, acc_pixel, acc_type

def train(model, train_data, train_labels):
    """
    Trains your model given the training data.
    """

    #TODO: Fix

    #Shuffling Input and Labels
    shuffled_indices = tf.random.shuffle(np.arange(train_labels.shape[0]))
    shuffled_train_inputs = tf.gather(train_data,shuffled_indices)
    shuffled_train_labels = tf.gather(train_labels,shuffled_indices)
    
    for i in range(0, len(train_data), model.batch_size):
        if(len(train_data)-i >= model.batch_size):
            # Batching:
            inputs = shuffled_train_inputs[i:i + model.batch_size, :, :, np.newaxis]
            labels = shuffled_train_labels[i:i+model.batch_size]
            
            #Backpropogation
            with tf.GradientTape() as tape:
                output_bb, output_pixel, output_type = model.call(inputs)
                loss = model.loss(output_bb, output_pixel, output_type, label_bb, label_pixel, label_type)

            #Deal with the Learning Rates Being Different Per Variable
            bias_variables_2_0 = []
            bias_variables_20_0 = []
            conv_var_1_1 = []
            conv_var_5_01 = []
            conv_var_20_01 = []
            bias_variables_2_0_var = []
            bias_variables_20_0_var = []
            conv_var_1_1_var = []
            conv_var_5_01_var = []
            conv_var_20_01_var = []
            for index,var in enumerate(model.trainable_variables):
                if "bias" in var.name:
                    if "conv8" in var.name:
                        bias_variables_20_0.append(index)
                        bias_variables_20_0_var.append(var)
                    else:
                        bias_variables_2_0.append(index)
                        bias_variables_2_0_var.append(var)
                else:
                    if "conv7" in var.name:
                        conv_var_5_01.append(index)
                        conv_var_5_01_var.append(var)
                    elif "conv8" in var.name:
                        conv_var_20_01.append(index)
                        conv_var_20_01_var.append(var)
                    else:
                        conv_var_1_1.append(index)
                        conv_var_1_1_var.append(var)
            
            #Gradient Descent
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer_bias_2_0.apply_gradients(zip(tf.gather(gradients, bias_variables_2_0), bias_variables_2_0_var))
            model.optimizer_bias_20_0.apply_gradients(zip(tf.gather(gradients, bias_variables_20_0), bias_variables_20_0_var))
            model.optimizer_conv_1_1.apply_gradients(zip(tf.gather(gradients, conv_var_1_1), conv_var_1_1_var))
            model.optimizer_conv_5_01.apply_gradients(zip(tf.gather(gradients, conv_var_5_01), conv_var_5_01_var))
            model.optimizer_conv_20_01.apply_gradients(zip(tf.gather(gradients, conv_var_20_01), conv_var_20_01_var))
        



def test(model, test_data, test_label):
    """
    Testing function for our model.
    """

    #TODO: Fix

    acc_bb_sum = 0
    acc_pixel_sum = 0
    acc_type_sum = 0
    number_of_batches = 0
    for i in range(0, len(test_data), model.batch_size):
        if(len(test_data)-i >= model.batch_size):

            # Batching:
            inputs = test_data[i:i + model.batch_size, :, :, np.newaxis]
            labels = test_label[i:i+model.batch_size]
            batch_molecules = test_data[i:i + model.batch_size]

            output_bb, output_pixel, output_type = model.call(inputs)
            acc_bb, acc_pixel, acc_type = model.accuracy_function(output_bb, output_pixel, output_type, label_bb, label_pixel, label_type)
            acc_bb_sum += acc_bb
            acc_pixel_sum += acc_pixel
            acc_type_sum += acc_type
            number_of_batches += 1
    
    acc_bb = acc_bb_sum / number_of_batches
    acc_pixel = acc_pixel_sum / number_of_batches
    acc_type = acc_type_sum / number_of_batches

    return acc_bb, acc_pixel, acc_type


def main():
    # Return the training and testing data from get_data
    train_data, train_label, test_data, test_label = preprocess.get_data('../../data/(TODO))')
    # Instantiate model
    model = TrafficSignModel()
    # Train and Test.
    for i in range(model.epochs):
        train(model, train_data)
    print(test(model, test_data))


if __name__ == '__main__':
    main()