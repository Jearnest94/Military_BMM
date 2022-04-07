import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None


def main():
    # Fixa cleaned_csv
    male_df = pd.read_csv('male_cleaned.csv')
    female_df = pd.read_csv('female_cleaned.csv')

    gender = input("male/female? ")
    weight = input("weight(kg): ")
    length = input("length(cm): ")
    weight = int(weight) * 10
    length = int(length) * 10
    print(f'You are', gender)

    if gender.lower() == 'male':
        gender = male_df
        add_tshirt_size(gender)
        add_pants_size_male(gender)
        calculate_k_10(gender, weight, length)
        sklearn_prediction(gender, weight, length)
        drawgraph(gender)

    else:
        gender = female_df
        add_tshirt_size(gender)
        add_pants_size_female(gender)
        calculate_k_10(gender, weight, length)
        sklearn_prediction(gender, weight, length)
        drawgraph(gender)


def add_tshirt_size(gender):
    tshirt_size = []
    tshirt_color = []

    for chest, waist in zip(gender['chestcircumference'], gender['waistcircumference']):
        if chest < 965:
            if waist >= 860:
                tshirt_size.append('Medium')
                tshirt_color.append('Orange')
            else:
                tshirt_size.append('Small')
                tshirt_color.append('Yellow')
        elif chest < 1040:
            if waist >= 915:
                tshirt_size.append('Large')
                tshirt_color.append('Red')
            else:
                tshirt_size.append('Medium')
                tshirt_color.append('Orange')
        elif chest < 1120:
            if waist >= 965:
                tshirt_size.append('X-Large')
                tshirt_color.append('Pink')
            else:
                tshirt_size.append('Large')
                tshirt_color.append('Red')
        elif chest < 1195:
            if waist >= 1015:
                tshirt_size.append('XX-Large')
                tshirt_color.append('Purple')
            else:
                tshirt_size.append('X-Large')
                tshirt_color.append('Pink')
        elif chest < 1270:
            tshirt_size.append('XX-Large')
            tshirt_color.append('Blue')
        elif chest < 1345:
            tshirt_size.append('XXX-Large')
            tshirt_color.append('Brown')
        elif chest < 1500:
            tshirt_size.append('XXXX-Large')
            tshirt_color.append('Green')

    gender['tshirt_size'] = tshirt_size
    gender['tshirt_color'] = tshirt_color


def add_pants_size_male(gender):
    pants_size = []
    pants_color = []

    for crotch, waist in zip(gender['crotchheight'], gender['waistcircumference']):
        if crotch < 810:
            if waist > 760:
                pants_size.append('Small')
                pants_color.append('Yellow')
            else:
                pants_size.append('X-Small')
                pants_color.append('White')
        elif crotch < 820:
            if waist > 840:
                pants_size.append('Medium')
                pants_color.append('Orange')
            else:
                pants_size.append('Small')
                pants_color.append('Yellow')
        elif crotch < 830:
            if waist > 920:
                pants_size.append('Large')
                pants_color.append('Red')
            else:
                pants_size.append('Medium')
                pants_color.append('Orange')
        elif crotch < 850:
            if waist > 1000:
                pants_size.append('X-Large')
                pants_color.append('Pink')
            else:
                pants_size.append('Large')
                pants_color.append('Red')
        elif crotch < 860:
            if waist > 1080:
                pants_size.append('XX-Large')
                pants_color.append('Purple')
            else:
                pants_size.append('X-Large')
                pants_color.append('Pink')
        elif crotch < 870:
            pants_size.append('XX-Large')
            pants_color.append('Purple')
        elif crotch < 880:
            pants_size.append('XXX-Large')
            pants_color.append('Blue')
        elif crotch >= 880:
            pants_size.append('XXXX-Large')
            pants_color.append('Brown')

    gender['pants_size'] = pants_size
    gender['pants_color'] = pants_color


def add_pants_size_female(gender):
    pants_size = []
    pants_color = []

    for butt, waist in zip(gender['buttockcircumference'], gender['waistcircumference']):
        if waist < 760:
            if butt >= 930:
                pants_size.append('Small')
                pants_color.append('Yellow')
            else:
                pants_size.append('X-Small')
                pants_color.append('White')
        elif waist < 840:
            if butt >= 960:
                pants_size.append('Medium')
                pants_color.append('Orange')
            else:
                pants_size.append('Small')
                pants_color.append('Yellow')
        elif waist < 920:
            if butt > 1020:
                pants_size.append('Large')
                pants_color.append('Red')
            else:
                pants_size.append('Medium')
                pants_color.append('Orange')
        elif waist < 1000:
            if butt > 1080:
                pants_size.append('X-Large')
                pants_color.append('Pink')
            else:
                pants_size.append('Large')
                pants_color.append('Red')
        elif waist < 1080:
            if butt > 1140:
                pants_size.append('XX-Large')
                pants_color.append('Purple')
            else:
                pants_size.append('X-Large')
                pants_color.append('Pink')
        elif waist >= 1080:
            if butt > 1200:
                pants_size.append('XXX-Large')
                pants_color.append('Blue')
            else:
                pants_size.append('XX-Large')
                pants_color.append('Purple')

    gender['pants_size'] = pants_size
    gender['pants_color'] = pants_color


def calculate_k_10(gender, weight, length):
    closest = []

    for data_length, data_weight in zip(gender['stature'], gender['weightkg']):
        delta_y = abs(length) - abs(data_length)
        delta_x = abs(weight) - abs(data_weight)
        hypo = delta_y ** 2 + delta_x ** 2
        closest.append(math.sqrt(hypo))

    gender['closest'] = closest
    K = gender.sort_values(by=['closest']).head(10)
    print(f'Recommended T-shirt size:', K['tshirt_size'].value_counts().idxmax())
    print(f'Recommended Pants size:', K['pants_size'].value_counts().idxmax())


def sklearn_prediction(gender, weight, length):
    x = np.array(list(zip(gender['stature'], gender['weightkg'])))
    y_ts = np.array(gender['tshirt_size'])
    y_ps = np.array(gender['pants_size'])
    x.reshape(1, -1)

    knn_ts = KNeighborsClassifier(n_neighbors=10)
    knn_ts.fit(x, y_ts)
    predicted_ts = knn_ts.predict([[length, weight]])

    knn_ps = KNeighborsClassifier(n_neighbors=10)
    knn_ps.fit(x, y_ps)
    predicted_ps = knn_ps.predict([[length, weight]])

    print(f'Sklearn KNeighborsClassifier suggests T-shirt size:', predicted_ts)
    print(f'Sklearn KNeighborsClassifier suggests Pants size:', predicted_ps)


def drawgraph(gender):
    sns.set()
    sns.scatterplot(y=gender['stature'], x=gender['weightkg'], c=gender['tshirt_color'])
    plt.show()


if __name__ == '__main__':
    main()
