import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

df = pd.read_csv('厦马职业50英语版.csv')

df.columns = ['Profession', 'Mathematical Skills', 'Programming Ability', 'Creativity', 'Analytical Skills',
              'Communication Skills', 'Leadership Skills', 'Business Acumen',
              'Problem-Solving', 'Teamwork', 'Adaptability']

df = df.dropna(how='all').reset_index(drop=True)
df = df[df['Profession'].str.strip().ne('')]

df.set_index('Profession', inplace=True)

df = df.astype(float)

colors = ["#8FB4BE", "#AFC9CF", "#D5E1E3", "#EBBFC2", "#E28187", "#D93F49"]
cmap = LinearSegmentedColormap.from_list("custom_heatmap", colors)

with PdfPages('profession_abilities_heatmap.pdf') as pdf:
    plt.figure(figsize=(16, 20))

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Ability Level'}
    )

    plt.title('Professional Abilities Heatmap', fontsize=16, pad=20)
    plt.xlabel('Abilities', fontsize=12)
    plt.ylabel('Profession', fontsize=12)

    plt.tight_layout(pad=3.0)
    pdf.savefig()
    plt.close()

print("profession_abilities_heatmap.pdf is saved.")