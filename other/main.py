import pandas as pd
from model_from_paper import EASE
'''
Runs the easer model made by the researchers, look at the implementation folder to see our model
'''
if __name__ == '__main__':
    interactions = pd.read_csv("../data/interactions_train.csv")
    # smaller=interactions.head(n=100)
    model=EASE()
    model.fit(interactions)
    print(model.predict())