import pandas as pd


def main():
    male_df = pd.read_csv('male.csv')
    male_cleaned = male_df[
        ['stature', 'weightkg', 'chestcircumference', 'waistcircumference', 'crotchheight', 'buttockcircumference']]
    male_cleaned.insert(4, column="tshirt_size", value="-")
    male_cleaned.insert(4, column="pants_size", value="-")
    male_cleaned.insert(5, column="tshirt_color", value="-")
    male_cleaned.insert(5, column="pants_color", value="-")

    female_df = pd.read_csv('female.csv')
    female_cleaned = female_df[
        ['stature', 'weightkg', 'chestcircumference', 'waistcircumference', 'crotchheight', 'buttockcircumference']]
    female_cleaned.insert(4, column="tshirt_size", value="-")
    female_cleaned.insert(4, column="pants_size", value="-")
    female_cleaned.insert(5, column="tshirt_color", value="-")
    female_cleaned.insert(5, column="pants_color", value="-")

    male_cleaned.to_csv('male_cleaned.csv', index=False)
    female_cleaned.to_csv('female_cleaned.csv', index=False)


if __name__ == '__main__':
    main()
