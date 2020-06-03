
from tensorflow.keras.layers import PReLU, Subtract, Add, Concatenate
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.initializers import VarianceScaling


class EBRNClass:
    def __init__(self, leaning_rate_dict, fine_tuning=False):
        """
        Construct the model class.
        :param leaning_rate_dict: learning rates and counterpart steps to change during training process
        :param fine_tuning: Boolean. train a model for first time (False) or fine tuning (True)
        of an already trained model
        """
        self.learning_rate_change = leaning_rate_dict
        self.lr_schedule = schedules.PiecewiseConstantDecay(boundaries=self.learning_rate_change['epoch'],
                                                            values=self.learning_rate_change['lr'])
        self.fine_tuning=fine_tuning

    def FeatureExt(self, input_data):
        """
        The first part of EBR Network, extract features from input image
        :param input_data: input image batch
        :return: tf object
        """
        x = input_data
        f = 256
        for i in range(3):
            x = Conv2D(filters=f, kernel_size=3, padding='same', activation=PReLU(),
                       kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                          distribution="untruncated_normal"),
                       name='FE_C{}'.format(str(i + 1)))(x)
            f = 64

        return x

    def BRModule(self, input_data, BRM_x, scale=4):
        """
        A single Block Residual Module (BRM)
        :param input_data: tf object
        :param BRM_x: index of BRM, x sub-index in the paper
        :param scale: magnifying scale factor
        :return: two tf objects for upper (super resolved) and lower (back-projected) flows
        """

        x1 = Conv2DTranspose(filters=64, kernel_size=scale, strides=scale, padding='valid', activation=PReLU(),
                             kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                                distribution="untruncated_normal"),
                             name='BRM{}_CT'.format(str(BRM_x)))(input_data)
        xup = x1
        for i in range(3):
            xup = Conv2D(filters=64, kernel_size=3, padding='same', activation=PReLU(),
                         kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                            distribution="untruncated_normal"),
                         name='BRM{}_C{}_u'.format(str(BRM_x), str(i + 1)))(xup)


        x2 = Conv2D(filters=64, kernel_size=scale, strides=scale, padding='valid', activation=PReLU(),
                    kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                       distribution="untruncated_normal"),
                    name='BRM{}_C{}_b'.format(str(BRM_x), str(1)))(x1)

        x2 = Subtract(name='BRM{}_S_b'.format(str(BRM_x)))([input_data, x2])
        xdn = x2

        for i in range(3):
            x2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=PReLU(),
                        kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                           distribution="untruncated_normal"),
                        name='BRM{}_C{}_b'.format(str(BRM_x), str(i + 2)))(x2)

        xdn = Add(name='BRM{}_A_b'.format(str(BRM_x)))([xdn, x2])
        return xup, xdn  # xup: SR flow in upper line,,, xdn: Residual flow in bottom line


    def EmbeddedBR(self, input_data, n_blocks, scale):
        """
        Combination of n BRMs
        :param input_data: tf object
        :param n_blocks: number of BRM in network
        :param scale: magnifying scale factor
        :return: tf object
        """
        x1 = []
        x2 = []
        # for the first BRM data comes from feature extraction layer
        xdn = input_data

        # execute all block residual modules (BRMs) by passing xdn from one to next BRM
        for i in range(0, n_blocks):
            xup, xdn = self.BRModule(xdn, BRM_x=i + 1, scale=scale)
            x1.append(xup)
            x2.append(xdn)

        # Add output of one BRM with output of its upper BRM then apply Conv2D
        for i in range(n_blocks - 1, 0, -1):
            x = Add(name='BRM{}_A_BRM{}'.format(str(i + 1), str(i)))([x1[i], x1[i - 1]])
            x1[i - 1] = Conv2D(filters=64, kernel_size=3, padding='same', activation=PReLU(),
                               kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                                  distribution="untruncated_normal"),
                               name='BRM{}_C'.format(str(i)))(x)

        # Concatenate all outputs of BRMs
        xup = x1[n_blocks - 1]
        for i in range(n_blocks - 2, -1, -1):
            xup = Concatenate(axis=-1, name='BRM{}_BRM{}_Co'.format(str(i + 2), str(i + 1)))([x1[i], xup])

        return xup

    def Reconstruct(self, input_data):
        """
        The last part of network to reconstruct the final image
        :param input_data: tf object
        :return: batch of super resolution images
        """
        # reconstruction layer
        x = Conv2D(filters=3, kernel_size=3, padding='same', activation=PReLU(),
                   kernel_initializer=VarianceScaling(scale=2.0, mode="fan_in",
                                                      distribution="untruncated_normal"),
                   name='Rec_C')(input_data)
        return x

    @staticmethod
    def normalize_01(img):
        """
        Normalise pixel values to the range of 0 to 1 (from 0 to 255)
        :param img: image array
        :return: normalised image array
        """
        return img / 255.0

    @staticmethod
    def denormalize_0255(img):
        """
        Denormalised pixel values to the range of 0 to 255
        :param img:
        :return:
        """
        return img * 255

    def create_model(self, number_of_blocks, scale_factor, LR_img_size, channel=3):
        """
        Compile the complete network as a keras model
        :param number_of_blocks: number of BRM units
        :param scale_factor: magnifying scale factor
        :param LR_img_size: size of input low res image normally 64
        :param channel: number of image channels, PNG image in RGB mode has 3 channels
        :return: keras model
        """
        input_LR = Input(shape=(LR_img_size, LR_img_size, channel), name='input_LR')
        x = Lambda(self.normalize_01)(input_LR)
        x = self.FeatureExt(x)
        x = self.EmbeddedBR(x, number_of_blocks, scale=scale_factor)
        x = self.Reconstruct(x)
        output_HR = Lambda(self.denormalize_0255, name='output_img')(x)

        model = Model(inputs=input_LR, outputs=output_HR, name='EBR_Net')

        if not self.fine_tuning:
            model.compile(optimizer=Adam(learning_rate=self.lr_schedule, epsilon=1e-08),
                          loss=mean_absolute_error,
                          metrics={'output_img': ['mse', 'accuracy']})
        else:
            model.compile(optimizer=Adam(learning_rate=self.lr_schedule, epsilon=1e-08),
                          loss=mean_squared_error,
                          metrics={'output_img': ['mae', 'accuracy']})

        return model
