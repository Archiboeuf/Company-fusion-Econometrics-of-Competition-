import pandas as pd
import numpy as np

df_param = pd.read_csv("F:/3A/S1/Econometrics of Competition/Projet/Data1/180107204404_ville300.csv", index_col=0)
df_part_marche=pd.read_csv("F:/3A/S1/Econometrics of Competition/Projet/Data1/parts_de_marche300.csv", index_col=0)

df = df_param
df_param.head()

for i in range(1,3):
    df["delta_%i"%i] = df["prix_%i"%i] - df["couts%i"%i]
    df["sucres%i_2"%i] = df["sucres%i"%i]**2

df[list(set(df.columns)-{"eta1","eta2","xi1","xi2"})].describe()

get_ipython().magic('pylab inline')
import seaborn as sns
import statsmodels.formula.api as sm

df_s=df_part_marche[["s0","s1","s2"]].reset_index()
df_s['ville']=df_s['ville'].apply(lambda x : x[1:]).astype(int) #on convertit en entier pour réordonner 
df_s_sorted=df_s.sort_values(by=["ville"]).reset_index()

base_comp=pd.concat([df_param,df_s_sorted],axis=1) #on ajoute les parts de marché à la base
base_comp=base_comp[["prix_1","prix_2","sucres1","sucres2","s0","s1","s2"]]
base_comp["sucres1_carre"]=base_comp["sucres1"].apply(lambda x : x**2) #on crée les variables sucre²
base_comp["sucres2_carre"]=base_comp["sucres2"].apply(lambda x : x**2)


base_comp['s1'][base_comp['s1']==0] #cas où la part de marché de l'entreprise 1 est nulle
len(base_comp)

#on retire les cas où il y a des parts de marché nulles
base_comp = base_comp[(base_comp['s1']!= 0 )& (base_comp['s2']!= 0 ) ] 
len(base_comp)

base_comp['l1'] = np.log(base_comp['s1']/base_comp['s0'])
base_comp['l2'] = np.log(base_comp['s2']/base_comp['s0'])

#On dédouble les données et on les remet en forme en rennomant 2 en 1 et vice versa dans l'une des bases
#pour les concaténer et avoir 1 : une entreprise et 2 : l'autre entreprise et interchanger les rôles
#C'est indispensable pour la régression plus loin
df1=base_comp.copy()
df2=base_comp.copy()
df2.rename(columns={'s1':'s2','s2':'s1','prix_1':'prix_2','prix_2':'prix_1','sucres1':'sucres2','sucres2':'sucres1',
                    'sucres1_carre':'sucres2_carre','sucres2_carre':'sucres1_carre','l1':'l2','l2':'l1'},inplace=True)

base_reg=pd.concat([df1,df2],ignore_index=True)

base_reg.head(5)


# ## Stat Desc

df_param.describe()

#Marge des entreprises
df_param['Marge_1'] = 100*df_param['delta_1']/df_param['prix_1']
df_param['Marge_2'] = 100*df_param['delta_2']/df_param['prix_2']


pd.DataFrame(df_param['prix_1'].describe())

pd.DataFrame(df_param['couts1'].describe())

#Cas où l'autre entreprise a une grande marge
df_param[df_param['Marge_2']>0.90].head()

for col in list(df_param):
    sns.distplot(df_param[col])
    plt.show()

sns.distplot(df_param['couts1'], label = "Côut", axlabel = False)
sns.distplot(df_param['prix_1'], label = "Prix", axlabel = False)
plt.legend();
plt.title("Comparaison entre les distributions de coûts de production et de prix de vente")
plt.show()

sns.distplot(df_param['couts2'], label = "Côut", axlabel = False)
sns.distplot(df_param['prix_2'], label = "Prix", axlabel = False)
plt.legend();
plt.title("Comparaison entre les distributions de coûts de production et de prix de vente")
plt.show()

col = 'Marge_1'
sns.distplot(df_param[col],label = "Marge en Pourcentage", axlabel = False)
plt.title("Marge de l'entreprise 1 pour toutes les boissons vendues")
plt.legend()
plt.show()

sns.boxplot(df_param[col])
plt.show()

plt.figure(figsize=[11,11])

sns.heatmap(df_param.corr(),annot=True,cmap="Oranges")

#Corrélations des variables avec la marge 1
df_param.corr()['Marge_1']


# ## Estimation IV2SLS

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from linearmodels.iv import IV2SLS

reg= IV2SLS.from_formula(' l1 ~ 1 + sucres1 + sucres1_carre + [prix_1 ~ sucres2+ sucres2_carre]', base_reg)
reg.fit()


# # 300 Villes
reg.fit()


# ## 600 Villes
reg.fit()


# ## 1000 Villes
reg.fit()

