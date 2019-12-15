import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

base = "/Users/liang/Works/KMSS/resource/"


def load_data(file_path, sep=","):
    line = 0
    row = []
    col = []
    data = []
    label = []
    with open(file_path, "r") as f:
        for rt in f:
            items = rt.strip().split(sep)
            label.append(int(items[0]))
            for vec, score in [i.split(":") for i in items[1:]]:
                if 4200 <= int(vec) < 4300:
                    continue
                row.append(line)
                col.append(int(vec))
                data.append(float(score))
            line += 1

    row_ind = np.array(row)
    # column indices
    col_ind = np.array(col)
    # data to be stored in COO sparse matrix
    data = np.array(data, dtype=float)
    mat_coo = coo_matrix((data, (row_ind, col_ind)))

    print(mat_coo.shape)
    print(mat_coo.data.size)

    return mat_coo, np.array(label)


def train_lr():
    x_train, y_train = load_data(base + "train_1.csv", sep=" ")
    lr = LogisticRegression(penalty='l1', C=1000.0, random_state=0, max_iter=110)
    lr.fit(x_train, y_train)

    print("training done")

    x_test, y_test = load_data(base + "test_2.csv", sep=" ")
    pred = lr.predict(x_test)
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and pred[i] > 0.5:
            correct += 1
        if y_test[i] == 0 and pred[i] < 0.5:
            correct += 1
    print("correct count ", correct, "accury", float(correct) / len(y_test))

    joblib.dump(lr, "./lr.model")
    pass


def xg_train():
    # dtrain = xgb.DMatrix(base + 'train_ctr_nosource.csv?format=csv&label_column=0')
    # dtest = xgb.DMatrix(base + 'test_ctr_nosource.csv?format=csv&label_column=0')
    dtrain = xgb.DMatrix(base + 'train_1.csv')
    dtest = xgb.DMatrix(base + 'test_2.csv')
    # specify parameters via map
    param = {'max_depth': 8, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] > 0.5:
            correct += 1
        if labels[i] == 0 and preds[i] < 0.5:
            correct += 1
    # print(preds[:100])
    # print(labels[:100])
    print(correct / 4000000.0)
    bst.dump_model('dump.raw.txt', 'featmap.txt')
    bst.save_model('001.model')


def load_xg_model():
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model('001.model')
    label = []
    row = []
    col = []
    data = []
    line = 0
    with open(base + 'test_ctr_1000.csv', 'r') as f:
        for rt in f:
            items = rt.strip().split(',')
            label.append(int(items[0]))
            for vec, score in [i.split(":") for i in items[1:]]:
                row.append(line)
                col.append(int(vec))
                data.append(float(score))
            line += 1

    # dtest = xgb.DMatrix(base + 'test_ctr_1000.csv?format=csv&label_column=0')
    # preds = bst.predict(dtest)
    # print(preds)
    # label = dtest.get_label()
    ddtest = xgb.DMatrix(csr_matrix((data, (row, col))))
    preds = bst.predict(ddtest)
    print(preds)
    correct = 0
    for i in range(len(label)):
        if label[i] == 1 and preds[i] > 0.5:
            correct += 1
        if label[i] == 0 and preds[i] < 0.5:
            correct += 1
    print("correct count ", correct, "accury", float(correct) / len(label))


def train_bd_short_video(db_base="/Users/liang/Downloads/"):
    dtrain = xgb.DMatrix(db_base + 'liang_test.train')
    dtest = xgb.DMatrix(db_base + 'liang_test.test')
    # dtrain = xgb.DMatrix(db_base + 'machine.txt.train')
    # dtest = xgb.DMatrix(db_base + 'machine.txt.test')
    # specify parameters via map
    param = {
        'booster': 'gblinear', 'eta': 0.5, 'objective': 'reg:squarederror',
        'save_period': 200, 'lambda': 0.000001,
        'alpha': 0.000001, 'updater': 'coord_descent', 'feature_selector': 'shuffle',
        'verbosity': 3, 'nthread': 8, "base_score": 0.3
    }
    num_round = 100
    # num_round = 4000
    bst = xgb.train(param, dtrain, num_round)
    print(bst.get_dump())
    # make prediction
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] > 0.83:
            correct += 1
        if labels[i] == 0 and preds[i] < 0.83:
            correct += 1
    print(preds[:100])
    # print(labels[:100])
    # print(correct / len(labels))
    bst.dump_model('db_dump3.raw.txt', 'featmap.txt')
    bst.save_model('db003.model')
    pass


def load_db_xg():
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model('db003.model')
    db_base = "/Users/liang/Downloads/"
    dtest = xgb.DMatrix(db_base + 'liang_test.test')
    preds = bst.predict(dtest)
    print(preds[:10])


def verify():
    bias = 0.103557
    weight = np.asarray([0,
                         -0.0106438,
                         0.0121984,
                         -0.0128787,
                         -0.0152616,
                         -0.00645582,
                         2.94731e-06,
                         -7.04664e-05,
                         9.30451e-06,
                         7.55125e-06,
                         -2.26425e-05,
                         2.7639e-06,
                         -9.29856e-05,
                         0.000123484,
                         -1.79572e-05,
                         5.56663e-05,
                         4.29742e-07,
                         -0.000115488,
                         0.0265938,
                         0.0100576,
                         -0.0255332,
                         0.0186512,
                         0.0233182,
                         0.00223656,
                         -0.00185646,
                         -0.000363419,
                         0.00153938,
                         -0.00190063,
                         -0.00119589,
                         -0.00322917,
                         0.0019237,
                         0.00507467,
                         0.000281712,
                         -0.00129153,
                         0.00405861,
                         0.0125834,
                         -0.00206373,
                         -0.00544814,
                         -0.0161974,
                         -0.00848895,
                         0.00493545,
                         0.00752835,
                         2.60131e-05,
                         0.00877803,
                         0.0566255,
                         -0.171239,
                         0.0654736,
                         -0.0675707,
                         0.096123,
                         0.000883401,
                         0.0853184,
                         -0.0110565,
                         0.177654,
                         0.28917,
                         -0.175487,
                         -0.00848339,
                         0.000149445,
                         -0.00700655,
                         -0.0112298,
                         -0.00303186,
                         -0.00151619,
                         0.0128071,
                         -0.000350444,
                         0.00194835,
                         -0.00633133,
                         0.00238764,
                         -0.000564223,
                         -0.00307746,
                         -0.00264047,
                         0.00355622,
                         0.00802713,
                         4.25148e-06,
                         0.123085,
                         -0.247486,
                         0.00889363,
                         0.0629471,
                         -0.0803144,
                         -0.0922555,
                         0.220114,
                         -0.044295,
                         0.0815077,
                         -0.258706,
                         0.00621451,
                         -0.00281427,
                         0.241881])

    with open("/Users/liang/Downloads/liang_test.test", 'r') as f:

        count = 0
        for line in f.readlines():
            line = line.strip("\n")
            items = line.split(" ")

            x = [0] * 85
            for item in items[1:]:
                pair = item.split(":")
                x[int(pair[0])] = float(pair[1])
            print(sum(weight * np.asarray(x)) + bias + 0.3)
            count += 1
            if count == 10:
                break


if __name__ == '__main__':
    train_bd_short_video()
    # verify()
    load_db_xg()

# if __name__ == '__main__':
#     xg_train()
#     pass
