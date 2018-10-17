import pandas as pd
from sdsj_feat import transform_categorical_features

_df = pd.DataFrame({
    'string_0': ['a', 'a', 'b'],
    'id_1': ['b', 'c', 'c'],
    'numeric_3': [1, 2, 3],
    'datetime_4': [None, None, None]
})


def test_cat_features():
    res = transform_categorical_features(_df)

    assert res['id_1']['c'] == 2
    assert res['string_0']['a'] == 2
