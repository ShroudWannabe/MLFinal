import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def pre(df):
    """
    Preprocess the data. This function should be used to clean and
    transform the data as needed. The function should return a pandas
    DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be cleaned.
    
    Returns
    -------
    df : pandas.DataFrame
        The cleaned dataset.
    """
    # Drop rows with missing values
    df = df.dropna()
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Drop rows with negative values
    df.drop('gameid', axis=1, inplace=True)
    df.drop('win', axis=1, inplace=True)
    df.drop('MIN', axis=1, inplace=True)
    df = df[df['season'] >= 2012]

   

    df.drop('FGM', axis=1, inplace=True)
    df.drop('FGA', axis=1, inplace=True)
    df.drop('FG%', axis=1, inplace=True)
    df.drop('3PM', axis=1, inplace=True)
    df.drop('3PA', axis=1, inplace=True)
    df.drop('FTM', axis=1, inplace=True)
    df.drop('FTA', axis=1, inplace=True)
    df.drop('FT%', axis=1, inplace=True)
    df.drop('OREB', axis=1, inplace=True)
    df.drop('DREB', axis=1, inplace=True)


    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['player', 'date'], inplace=True)

    # Compute rolling averages and reset index after
    avg_last_5 = df.groupby('player')['PTS'].rolling(window=5, min_periods=1).mean().reset_index()
    avg_last_20 = df.groupby('player')['PTS'].rolling(window=20, min_periods=1).mean().reset_index()

    # Rename 'PTS' column to match the intended output
    avg_last_5 = avg_last_5.rename(columns={'PTS': 'Avg_Last_5_Games'})
    avg_last_20 = avg_last_20.rename(columns={'PTS': 'Avg_Last_20_Games'})

    # Merge the rolling averages back into the original DataFrame
    df = df.merge(avg_last_5[['level_1', 'Avg_Last_5_Games']], left_index=True, right_on='level_1', how='left').drop(columns=['level_1'])
    df = df.merge(avg_last_20[['level_1', 'Avg_Last_20_Games']], left_index=True, right_on='level_1', how='left').drop(columns=['level_1'])

    # Continue with one-hot encoding and other preprocessing steps...
    df = one_hot_encode_columns(df, ['type', 'away', 'team', 'home'])

    df['Month'] = df['date'].dt.month
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df.drop('date', axis=1, inplace=True)
    
    return df




def one_hot_encode_columns(df, columns):
    """
    One-hot encode specified columns and add them back to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be processed.
    columns : list of str
        The column names to be one-hot encoded.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the specified columns one-hot encoded.
    """
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame")
        
        if df[column].nunique() > 20:
            print(f"Warning: '{column}' has more than 20 unique values. Skipping one-hot encoding.")
            continue
        
        encoder = OneHotEncoder(sparse_output=False)
        encoded_column = encoder.fit_transform(df[[column]])
        feature_names = encoder.get_feature_names_out([column])
        encoded_df = pd.DataFrame(encoded_column, columns=feature_names, index=df.index)
        
        df = df.drop(columns=[column]).join(encoded_df)
    
    return df


'''
    # Drop rows with negative values
    df.drop('gameid', axis=1, inplace=True)
    df.drop('win', axis=1, inplace=True)
    df.drop('MIN', axis=1, inplace=True)
    df = df[df['season'] >= 2012]

   

    df.drop('FGM', axis=1, inplace=True)
    df.drop('FGA', axis=1, inplace=True)
    df.drop('FG%', axis=1, inplace=True)
    df.drop('3PM', axis=1, inplace=True)
    df.drop('3PA', axis=1, inplace=True)
    df.drop('FTM', axis=1, inplace=True)
    df.drop('FTA', axis=1, inplace=True)
    df.drop('FT%', axis=1, inplace=True)
    df.drop('OREB', axis=1, inplace=True)
    df.drop('DREB', axis=1, inplace=True)
'''


'''
    if 'type' not in df.columns:
        raise KeyError("Column 'type' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['type']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['type'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)


    if 'away' not in df.columns:
        raise KeyError("Column 'away' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['away']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['away'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)


    if 'team' not in df.columns:
        raise KeyError("Column 'team' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['team']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['team'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)


    if 'home' not in df.columns:
        raise KeyError("Column 'home' does not exist in the DataFrame")
    encoder = OneHotEncoder()
    encoded_column = encoder.fit_transform(df[['home']])
    feature_names = encoder.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_column.toarray(), columns=feature_names)
    df.drop(columns=['home'], inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    
    df['Month'] = df['date'].dt.month
    df['DayOfWeek'] = df['date'].dt.dayofweek

    
    df = df.drop(['date'], axis=1)

    return df
'''




def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="filename of training data")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = pre(df)
    train_df, test_df = train_test_split(df)
    # solution here

    
    nonext = os.path.splitext(args.input)[0]
    print("Training DF Shape:", train_df.shape)
    train_df.to_csv(nonext+"_train.csv", index=False)
    print("Test DF Shape:", test_df.shape)
    test_df.to_csv(nonext+"_test.csv", index=False)
    print(f"Training DF Shape: {train_df.shape}")
    print(f"Test DF Shape: {test_df.shape}")

if __name__ == "__main__":
    main()
