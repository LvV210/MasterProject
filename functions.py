import pandas as pd

# def scientific_notation(dataframe):
#     # Create a copy of the DataFrame to avoid modifying the original data
#     formatted_df = dataframe.copy()

#     # Iterate through the columns and format numerical values in scientific notation
#     for col in formatted_df.columns:
#         if pd.api.types.is_numeric_dtype(formatted_df[col]):
#             formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2E}")

#     return formatted_df

def scientific_notation(df):
    df2 = df.copy()
    # Get the number of rows and columns in the DataFrame
    num_rows, num_columns = df2.shape

    # Double for loop to iterate through all entries
    for i in range(num_rows):
        for j in range(num_columns):
            # Check if var1 is a number
            if isinstance(df2.loc[i,j], (int, float, complex)):
                df2.loc[i,j] = "{:.2E}".format(df2.loc[i,j])

    return df2