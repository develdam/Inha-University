import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset("titanic")
titanic.to_csv("/content/drive/MyDrive/Colab Notebooks/titanic.csv", index=False)

titanic["age"].fillna(28, inplace=True)
titanic["embarked"].fillna("S", inplace=True)
titanic["deck"].fillna("C",inplace=True)
titanic["embark_town"].fillna("Southampton",inplace=True)

titanic.groupby(["survived","sex"])["survived"].count()

plt.pie([len(titanic.loc[(titanic["survived"]==0)&(titanic["sex"]=="male")]),len(titanic.loc[(titanic["survived"]==0)&(titanic["sex"]=="female")]),\
len(titanic.loc[(titanic["survived"]==1)&(titanic["sex"]=="male")]),len(titanic.loc[(titanic["survived"]==1)&(titanic["sex"]=="female")])],\
labels=["non-survived male","non-survived female","survived male","survived female"],autopct="%.2f")

plt.title("Non-Survivor by Pclass")
plt.bar(["1","2","3"],[len(titanic.loc[(titanic["pclass"]==1)&(titanic["survived"]==0)]),len(titanic.loc[(titanic["pclass"]==2)&(titanic["survived"]==0)]),len(titanic.loc[(titanic["pclass"]==3)&(titanic["survived"]==0)])])

plt.title("Survivor by Pclass")
plt.bar(["1","2","3"],[len(titanic.loc[(titanic["pclass"]==1)&(titanic["survived"]==1)]),len(titanic.loc[(titanic["pclass"]==2)&(titanic["survived"]==1)]),len(titanic.loc[(titanic["pclass"]==3)&(titanic["survived"]==1)])])

titanic.corr(method="pearson")

sns.pairplot(titanic.corr())

sns.pairplot(titanic)

titanic = pd.get_dummies(titanic, columns=["sex"], drop_first=True)
sns.heatmap(titanic.corr(),cmap='Reds',annot=True)
