import pandas as pd
from model import EASE
if __name__ == '__main__':
    interactions = pd.read_csv("data/interactions_test.csv")
    smaller=interactions.head(n=100)
    model=EASE()
    model.fit(smaller)
    print(model.predict())