# based on code from https://github.com/saravrajavelu/Adult-Income-Analysis/blob/master/Adult_Income_Analysis.ipynb

import pandas as pd
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
continous_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain','capital-loss', 'hours-per-week']


def normalize(df):
    result = df.copy()
    for feature_name in continous_columns:
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean) / (std)
    return result

def to_one_hot(df, df_cols):
    df_1 = df.drop(columns=df_cols, axis=1)
    df_2 = pd.get_dummies(df[df_cols])
    return pd.concat([df_1, df_2], axis=1, join='inner')


if __name__ == "__main__":
    train = pd.read_csv('adult.data', sep=",\s", header=None, names = column_names, engine = 'python')
    test = pd.read_csv('adult.test', sep=",\s", header=None, names = column_names, engine = 'python')
    test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
    print(len(train), len(test))

    adult = pd.concat([test,train])
    adult['income'] = adult['income'].astype('category').cat.codes

    adult = to_one_hot(adult, categorical_columns)
    adult = normalize(adult)
    adult_test = adult[:len(test)]
    adult_train = adult[len(test):]
    adult_train.to_pickle('adult_train_1h.pickle')
    adult_test.to_pickle('adult_test_1h.pickle')
