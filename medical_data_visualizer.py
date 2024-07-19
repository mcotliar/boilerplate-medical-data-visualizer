import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight']/((df['height']/100) *
                    (df['height']/100)) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
# 4


def draw_cat_plot():

    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[
                     "cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # 6
    df_cat = df_cat = df_cat[['cardio', 'variable', 'value']].groupby(
        ['cardio', 'variable']).value_counts().reset_index()

    # 7
    graph = sns.catplot(data=df_cat, kind='count',  x='variable',
                        hue='value', col='cardio').set(ylabel='total').figure

    # 8
    fig = graph.figure

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 14
    fig, ax = plt.subplots()
    # 15
    ax = sns.heatmap(corr, fmt='0.1f', annot=True, mask=mask)

    # 16
    fig.savefig('heatmap.png')
    return fig
