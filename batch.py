import copy

import numpy as np

from util import remove_item, add_mask


def get_train_data(dataset, num_negatives):
    user_input, item_input, labels, batch_length = [], [], [], []
    train_list = dataset.trainList
    for u in range(dataset.num_users):
        if u == 0:
            batch_length.append((1 + num_negatives) * len(train_list[u]))
        else:
            batch_length.append((1 + num_negatives) * len(train_list[u]) + batch_length[u - 1])
        for i in train_list[u]:
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)

            # sample distinct negative instances for per positive instance
            item_list = []
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(dataset.num_items)
                while j in train_list[u] or j in item_list:
                    j = np.random.randint(dataset.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)

                item_list.append(j)
    return user_input, item_input, labels, batch_length


def get_batch_train_data(batch_id, train_data, train_list, num_items):
    user_input, item_input, labels, batch_length = train_data[0], train_data[1], \
                                                   train_data[2], train_data[3]
    # represent the feature of users via items rated by him/her
    user_list, num_list, item_list, labels_list = [], [], [], []
    if batch_id == 0:
        begin = 0
    else:
        begin = batch_length[batch_id - 1]
    batch_index = list(range(begin, batch_length[batch_id]))
    np.random.shuffle(batch_index)
    for idx in batch_index:
        user_idx = user_input[idx]
        item_idx = item_input[idx]

        # items rated by user
        rated_items = []
        rated_items.extend(train_list[user_idx])

        num_list.append(remove_item(num_items, rated_items, item_idx))
        user_list.append(rated_items)
        item_list.append(item_idx)
        labels_list.append(labels[idx])
    user_input = np.array(add_mask(num_items, user_list, max(num_list)))
    num_idx = np.array(num_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return user_input, num_idx, item_input, labels


def get_batch_test_data(batch_id, dataset):
    train_list = dataset.trainList
    rating = dataset.testRatings[batch_id]

    u, test_item = rating[0], rating[1]

    assert u == batch_id

    n_u = len(train_list[u])

    item_list = copy.deepcopy(dataset.testNegatives[batch_id])
    item_list.append(test_item)

    user_list = [train_list[u]] * len(item_list)

    return np.array(user_list), np.array(item_list), test_item, n_u
