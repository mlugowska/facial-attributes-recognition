import os
import pandas as pd
import matplotlib.pyplot as plt


# Logistic Regression: 23 features selected


class CelebA():
    '''Wraps the celebA dataset, allowing an easy way to:
         - Select the features of interest,
         - Split the dataset into 'training', 'test' or 'validation' partition.
    '''

    def __init__(self, main_folder='celeba-dataset/', selected_features=None, drop_features=[]):
        self.main_folder = main_folder
        self.images_folder = os.path.join(main_folder, 'img_align_celeba/')
        self.attributes_path = os.path.join(main_folder, 'list_attr_celeba.csv')
        self.partition_path = os.path.join(main_folder, 'list_eval_partition.csv')
        self.selected_features = selected_features
        self.features_name = []
        self.__prepare(drop_features)

    def __prepare(self, drop_features):
        '''do some preprocessing before using the data: e.g. feature selection'''
        # attributes:
        if self.selected_features is None:
            self.attributes = pd.read_csv(self.attributes_path)
            self.num_features = 40
        else:
            self.num_features = len(self.selected_features)
            self.selected_features = self.selected_features.copy()
            self.selected_features.append('image_id')
            self.attributes = pd.read_csv(self.attributes_path)[self.selected_features]

        # remove unwanted features:
        for feature in drop_features:
            if feature in self.attributes:
                self.attributes = self.attributes.drop(feature, axis=1)
                self.num_features -= 1

        self.attributes.set_index('image_id', inplace=True)
        self.attributes.replace(to_replace=-1, value=0, inplace=True)
        self.attributes['image_id'] = list(self.attributes.index)

        self.features_name = list(self.attributes.columns)[:-1]

        # load ideal partitioning:
        self.partition = pd.read_csv(self.partition_path)
        self.partition.set_index('image_id', inplace=True)

    def split(self, name='training', drop_zero=False):
        '''Returns the ['training', 'validation', 'test'] split of the dataset'''
        # select partition split:
        if name is 'training':
            to_drop = self.partition.where(lambda x: x != 0).dropna()
        elif name is 'validation':
            to_drop = self.partition.where(lambda x: x != 1).dropna()
        elif name is 'test':  # test
            to_drop = self.partition.where(lambda x: x != 2).dropna()
        else:
            raise ValueError('CelebA.split() => `name` must be one of [training, validation, test]')

        partition = self.partition.drop(index=to_drop.index)

        # join attributes with selected partition:
        joint = partition.join(self.attributes, how='inner').drop('partition', axis=1)

        if drop_zero is True:
            # select rows with all zeros values
            return joint.loc[(joint[self.features_name] == 1).any(axis=1)]
        elif 0 <= drop_zero <= 1:
            zero = joint.loc[(joint[self.features_name] == 0).all(axis=1)]
            zero = zero.sample(frac=drop_zero)
            return joint.drop(index=zero.index)

        return joint


# load the dataset with 37 out of 40 features:
# celeba = CelebA(selected_features=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bald', 'Big_Lips', 'Big_Nose',
#                                    'Black_Hair', 'Blond_Hair', 'Blurry', 'Bushy_Eyebrows', 'Double_Chin',
#                                    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Mustache',
#                                    'No_Beard', 'Receding_Hairline', 'Sideburns', 'Wearing_Earrings',
#                                    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'])

celeba = CelebA()

# couting number of samples per partition:
print(f"total entries: {celeba.attributes.shape[0]}")
print(f"  - training: {celeba.split('training').shape[0]}")
print(f"  - validation: {celeba.split('validation').shape[0]}")
print(f"  - test: {celeba.split('test').shape[0]}")

# shows five random samples
celeba.attributes.sample(5)

# ------------------------------------------------------------------------------
# -- Computing sorted features frequency on all the 202k entries
# ------------------------------------------------------------------------------

frequencies = celeba.attributes.mean(axis=0).sort_values()

_ = frequencies.plot(title='CelebA Attributes Frequency',
                     kind='barh',
                     figsize=(12, 12),
                     color='b')

# ------------------------------------------------------------------------------
# -- Shows a random sample image with its attributes:
# ------------------------------------------------------------------------------
#
# import cv2
#
# # pick a sample
# random_sample = celeba.attributes.sample(1)
# pic_path = os.path.join(celeba.images_folder, random_sample.index[0])
#
# print(f"path: '{pic_path}'")
# print(f"attributes: {random_sample.to_numpy()[0, :-1]}")
#
# # load and convert the image
# pic = cv2.imread(pic_path)
# pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
#
# # plot with attributes
# plt.imshow(pic)
# plt.grid(False)
plt.show()

# random_sample.drop('image_id', axis=1)
