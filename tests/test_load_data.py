from sdsj_feat import load_data


PATH_TO_TRAIN = '../data/check_1_r/train.csv'
PATH_TO_TEST = '../data/check_1_r/test.csv'


def test_train():
    df, y, model_config, line_id = load_data(PATH_TO_TRAIN)

    assert 1 == 1