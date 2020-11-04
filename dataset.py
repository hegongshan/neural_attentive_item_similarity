"""
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
"""
import scipy.sparse as sp
import numpy as np
import os

ITEM_CLIP = 300


class DataSet(object):

    def __init__(self, path, data_set_name):
        file_path = os.path.join(path, data_set_name)
        self.trainMatrix = self.load_rating_file_as_matrix(file_path + ".train.rating")
        self.trainList = self.load_training_file_as_list(file_path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(file_path + ".test.rating")
        self.testNegatives = self.load_negative_file(file_path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    @staticmethod
    def load_rating_file_as_list(filename):
        rating_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                rating_list.append([user, item])
                line = f.readline()
        return rating_list

    @staticmethod
    def load_negative_file(filename):
        negative_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = list(map(int, arr[1:]))
                negative_list.append(negatives)
                line = f.readline()
        return negative_list

    @staticmethod
    def load_rating_file_as_matrix(filename):
        """
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        """
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    @staticmethod
    def load_training_file_as_list(filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as file:
            for line in file:
                if line is not None and line != "":
                    arr = line.split("\t")
                    u, i = int(arr[0]), int(arr[1])
                    if u_ == u:
                        if len(items) < ITEM_CLIP:
                            items.append(i)
                    else:
                        lists.append(items)
                        items = []
                        u_ += 1
        lists.append(items)
        print("already load the trainList...")
        return lists


if __name__ == '__main__':
    path = 'data'
    data_set = 'ml-1m'
    data = DataSet(path, data_set)
    print(data.trainList[0])
