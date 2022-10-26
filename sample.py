import numpy as np


def sample_uniform(dataset):
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_user, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.n_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])

    return np.array(S)
