def load_data(file_location):

    import pandas as pd

    data = pd.read_csv(file_location, sep= ",", header=0)

    return data