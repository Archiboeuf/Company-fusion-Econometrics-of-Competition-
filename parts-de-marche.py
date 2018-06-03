from __future__ import division
import random as rd
import math
import pandas as pd
import numpy as np
import scipy
import time
from scipy.stats import beta, uniform, gamma
from scipy.optimize import fsolve
#from pathos.multiprocessing import ProcessingPool as Pool
import numba
import datetime
import argparse

def cdf_extreme_value(x):
    return(np.exp(-np.exp(-x)))
def inverse_cdf_extreme_value(x):
    return(-np.log(-np.log(x)))
    
nb_villes=300
alpha,beta1,beta2,gam=1,2,1,1
nb_consommateurs=1000


# Question 1
def main():
    parser = argparse.ArgumentParser() #définition des arguments
    parser.add_argument("--nb_villes", default=1000, type=int)
    parser.add_argument("--alpha", default=1,type=int)
    parser.add_argument("--beta1", default=2,type=int)
    parser.add_argument("--beta2", default=1,type=int)
    parser.add_argument("--gam", default=1, type=int)

    args = parser.parse_args()
    now = datetime.datetime.now()
    now = now.strftime("%y%m%d%H%M%S_")

    nb_villes=args.nb_villes
    a = 2
    b = 3
    nb_entreprises=3 
    nb_consommateurs=1000
    alpha= args.alpha
    beta1= args.beta1
    beta2= args.beta2
    gam = args.gam
    sucres1=beta.rvs(a,b,size=nb_villes)
    sucres2=beta.rvs(a,b,size=nb_villes)

    a=0.5
    eta1=gamma.rvs(a=0.05, scale=2, size=nb_villes)
    eta2=gamma.rvs(a=0.05, scale=2, size=nb_villes)

    def epsilons_utilite(nb_villes,nb_consommateurs,nb_entreprises,inverse_cdf): #simulations des epsilons
        entreprises=['j'+str(i) for i in range(nb_entreprises)] 
        villes=['v'+str(i) for i in range(nb_villes) for j in entreprises]
        entreprises=entreprises*nb_villes
        rand = np.random.uniform(0,1,(nb_consommateurs,nb_villes*nb_entreprises))
        epsilons=pd.DataFrame(data=inverse_cdf(rand),
                              columns=pd.MultiIndex.from_tuples([z for z in zip(villes,entreprises)]))
        return(epsilons) 

    epsilons=epsilons_utilite(nb_villes,nb_consommateurs,nb_entreprises,inverse_cdf_extreme_value)

    xi1=uniform.rvs(size=nb_villes)
    xi1=inverse_cdf_extreme_value(xi1)

    xi2=uniform.rvs(size=nb_villes)
    xi2=inverse_cdf_extreme_value(xi2)

    couts1=a*sucres1+eta1
    couts2=a*sucres2+eta2

    def condition(k,prix,sucres, xis, couts): #condition d'équilibre de Nash
        delt = lambda j : alpha + beta1*sucres[j]-beta2*sucres[j]**2-gam*prix[j]+xis[j]
        exp_delta = lambda j : np.exp(delt(j))
        pdm = lambda j : (exp_delta(j))/(1+exp_delta(0)+exp_delta(1))
        return 1+gam*(prix[k]-couts[k])*(pdm(k)-1)
    
    def jacobian(k,j, prix, sucres, xis, couts): #Implémentation exacte du jacobien
        delt = lambda j : alpha + beta1*sucres[j]-beta2*sucres[j]**2-gam*prix[j]+xis[j]
        exp_delta = lambda j : np.exp(delt(j))
        pdm = lambda j : (exp_delta(j))/(1+exp_delta(0)+exp_delta(1))
        if k != j:
            term = ((np.exp(-delt(j))+np.exp(delt(j)-delt(k)))*(prix[k]-couts[k]))/(1+np.exp(delt(j))+np.exp(delt(k)))**2
            return gam*(-1.+pdm(k)-gam*term)
        else:
            return gam**2*(prix[k]-couts[k])*((np.exp(delt(j)+delt(k)))/(1+np.exp(delt(j))+np.exp(delt(k)))**2)

    def multi_jacobian(prix,sucres, xis, couts):
        params = [sucres, xis, couts]
        return(
        [
                [jacobian(i,j, prix, *params) for j in range(2)] for i in range(2)
            ]
        )
    
    def conditions(prix, sucres, xis, couts): # les deux conditions de l'équilibre
        return([condition(0, prix, sucres, xis, couts),
                condition(1, prix, sucres, xis, couts)])

    def prix_equilibre(sucres1,sucres2,xi1,xi2,couts1,couts2,ville): # résolution numérique de l'équilibre
        sucres=[sucres1[ville], sucres2[ville]]
        xis=[xi1[ville],xi2[ville]]
        couts=[couts1[ville],couts2[ville]]
        result = (scipy.optimize.fsolve(lambda x: conditions(x, sucres, xis, couts),
                                     [couts[0],couts[1]],fprime=lambda x: multi_jacobian(x, sucres, xis, couts) )
            )
        result_cleaned = [p if p > 0 else couts[n] for n,p in enumerate(result)]
        return(result_cleaned)

    #p = Pool(10)
    #f = lambda v: prix_equilibre(sucres1,sucres2,xi1,xi2,couts1,couts2,v) 
    #prix = p.map(f, range(nb_villes))
    prix = [prix_equilibre(sucres1,sucres2,xi1,xi2,couts1,couts2,v) for v in range(nb_villes)]

    #exportation des données
    df_prix = pd.DataFrame(prix, columns=["prix_1","prix_2"])


    df_param = pd.DataFrame(np.array([eta1,eta2,xi1,xi2,sucres1,sucres2,couts1,couts2]).T,
                 columns=["eta1","eta2","xi1","xi2","sucres1","sucres2","couts1","couts2"])

    pd.concat([df_prix,df_param],axis=1).to_csv(now+'ville'+str(nb_villes)+'.csv')

    epsilon_df = epsilons.T.reset_index()

    df_param["ville"] = ["v%i"%a for a in range(nb_villes)]

    epsilon_df.columns = ["ville", "produit"]+["conso_%i"%a for a in range(1000)]
    epsilon_df.merge(df_param, on="ville").to_csv(now+"total_ville"+str(nb_villes)+".csv")

if __name__ == "__main__":
    main()
    
#chargement de données sauvegardées
ville300=pd.read_csv('C:\\Users\\benji\\Desktop\\Cours\\Econometrics of competition\\180107204404_total_ville'+str(nb_villes)+'.csv',
                     index_col=0)
#mise en forme des données
eta1 = ville300[ville300["produit"] =="j1"]["eta1"].reset_index(drop=True)
eta2 = ville300[ville300["produit"] =="j2"]["eta2"].reset_index(drop=True)
xi1 = ville300[ville300["produit"] =="j1"]["xi1"].reset_index(drop=True)
xi2 = ville300[ville300["produit"] =="j2"]["xi2"].reset_index(drop=True)
sucres1 = ville300[ville300["produit"] =="j1"]["sucres1"].reset_index(drop=True)
sucres2 = ville300[ville300["produit"] =="j2"]["sucres2"].reset_index(drop=True)
couts1 = ville300[ville300["produit"] =="j1"]["couts1"].reset_index(drop=True)
couts2 = ville300[ville300["produit"] =="j2"]["couts2"].reset_index(drop=True)


ville300bis=pd.read_csv('C:\\Users\\benji\\Desktop\\Cours\\Econometrics of competition\\180107204404_ville'+ str(nb_villes) +'.csv',
                     index_col=0)

#toujours de la mise en forme
prix=ville300bis[['prix_1','prix_2']]
prix1=ville300bis['prix_1']
prix2=ville300bis['prix_2']
ville300bis['j1']='j1'
ville300bis['j2']='j2'
ville300bis.reset_index(inplace=True)
ville300bis['ville']='v'+ville300bis['index'].astype(str)

villes=ville300.copy()
villes['eta']=villes['eta1']*(villes['produit']=='j1')+villes['eta2']*(villes['produit']=='j2')
villes['xi']=villes['xi1']*(villes['produit']=='j1')+villes['xi2']*(villes['produit']=='j2')
villes['sucres']=villes['sucres1']*(villes['produit']=='j1')+villes['sucres2']*(villes['produit']=='j2')
villes['couts']=villes['couts1']*(villes['produit']=='j1')+villes['couts2']*(villes['produit']=='j2')
villes=pd.merge(villes,ville300bis[['prix_1','ville','j1']],how='outer',
                  left_on=['ville','produit'],right_on=['ville','j1'])
villes=pd.merge(villes,ville300bis[['prix_2','ville','j2']],how='outer',
                  left_on=['ville','produit'],right_on=['ville','j2'])
villes[['prix_1','prix_2']]=villes[['prix_1','prix_2']].fillna(0)
villes['prix']=villes['prix_1']*(villes['produit']=='j1')+villes['prix_2']*(villes['produit']=='j2')


def calculer_utilite2(villes): #calcul de l'utilité (version 1 obsolète supprimée)
    utilite=villes.copy()
    for i in range(nb_consommateurs):
        col='conso_'+str(i)
        utilite[col]=alpha+beta1*utilite['sucres']-beta2*utilite['sucres']**2-gam*utilite['prix']+utilite[col]
    return(utilite)

utilite=calculer_utilite2(villes)

#on cherche les parts de marché et donc les boissons consommées
utilite.groupby('ville')[['produit','conso_0']].apply(lambda x : x['produit'][x['conso_0'].argmax()])
utilite['produit'][utilite[utilite['ville']=='v0'][['conso_0','conso_3']].apply(pd.Series.argmax)]

res=pd.DataFrame(utilite.groupby('ville')[['produit','conso_0']].apply(lambda x : x['produit'][x['conso_0'].argmax()]),
                 columns=['conso_0'])
for i in range(1, nb_consommateurs):
    col='conso_'+str(i)
    res[col]=utilite.groupby('ville')[['produit',col]].apply(lambda x : x['produit'][x[col].argmax()])

transp=res.transpose()
res['s0']=0
res['s1']=0
res['s2']=0

for col in transp.columns:
    temp=transp[col].value_counts()
    if 'j0' in temp:
        res.loc[col,'s0']=temp['j0']/1000
    if 'j1' in temp:
        res.loc[col,'s1']=temp['j1']/1000
    if 'j2' in temp:
        res.loc[col,'s2']=temp['j2']/1000

#exportation de données
#res.to_csv('C:\\Users\\benji\\Desktop\\Cours\\Econometrics of competition\\parts de marche'+str(nb_villes)+'.csv')
#utilite.to_csv('C:\\Users\\benji\\Desktop\\Cours\\Econometrics of competition\\utilite '+str(nb_villes)+'.csv')
res.reset_index(inplace=True)


### Questions 5 ###

#estimateurs obtenus dans la question 3 pour 300 villes
alpha_hat=0.1524
beta1_hat=2.0809
beta2_hat=0.9675
gam_hat=1.1052

ville300ter=pd.merge(ville300bis,res[['ville','s0','s1','s2']],how='outer',on='ville') #on ajoute les parts de marché

ville300ter['couts1_hat']=ville300ter['prix_1']-1/(gam_hat*(1-ville300ter['s1'])) #couts estimés
ville300ter['couts2_hat']=ville300ter['prix_2']-1/(gam_hat*(1-ville300ter['s2']))


def log(x): #problème de log quand la part de marché est nulle
    if x<=0:
        return(np.nan)
    else:
        return(math.log(x))
for i in ['1','2']:
    ville300ter['xi'+i+'_hat']=ville300ter[['s'+i,'s0','sucres'+i,'prix_'+i]].apply(lambda x : 
        log(x['s'+i]/x['s0'])-alpha_hat-beta1_hat*x['sucres'+i]+beta2_hat*x['sucres'+i]**2+gam_hat*x['prix_'+i],
        axis=1)
    ville300ter['xi'+i+'_hat'].fillna(ville300ter['xi'+i+'_hat'].mean(),inplace=True) #nan si s=0
ville300ter['xi0_hat']=[-alpha_hat]*nb_villes

#on prépare des fonctions pour simuler l'erreur pour l'eau du robinet
def simuler_extreme_cond(util,cond='inf'): #si inf envoie un x~extremevalue sachant x<util sinon sachant x>util
    if cond=='inf':
        x=rd.uniform(0,cdf_extreme_value(util))
    else:
        x=rd.uniform(cdf_extreme_value(util),1)
    return(inverse_cdf_extreme_value(x))

def simuler_erreur(utilite,col):
    conso=utilite[col]
    utilite[col+'_j0']=[0]*nb_villes
    utilite.loc[conso=='j0',col+'_j0']=utilite[conso=='j0'].apply(lambda x : 
        simuler_extreme_cond(max(x[col+'_j1'],x[col+'_j2']),'sup'),axis=1)
    utilite.loc[conso!='j0',col+'_j0']=utilite[conso!='j0'].apply(lambda x : 
        simuler_extreme_cond(max(x[col+'_j1'],x[col+'_j2']),'inf'),axis=1)
    return(utilite)


def calculer_utilite3(villes,res):
    vil=villes.copy()
    for col in ['s0','s1','s2']:
        del vil[col]
    utilite=pd.merge(vil,res,how='outer',on='ville')
    for i in range(nb_consommateurs):
        col='conso_'+str(i)
        utilite[col+'_j1']=(alpha_hat + 
           (beta1_hat*utilite['sucres1']-beta2_hat*utilite['sucres1']**2-
            gam_hat*utilite['prix_1']+utilite['xi1_hat']))
        utilite[col+'_j2']=(alpha_hat+
           (beta1_hat*utilite['sucres2']-beta2_hat*utilite['sucres2']**2-
              gam_hat*utilite['prix_2']+utilite['xi2_hat']))
        utilite=simuler_erreur(utilite,col) #simule l'utilite de l'eau
    return(utilite)
    
utilite=calculer_utilite3(ville300ter,res)
utilite=utilite[['conso_'+str(i)+'_j'+j for i in range(nb_consommateurs) for j in ['0','1','2']]]

#utilité de la boisson choisie
utilite_choisie=utilite.copy()
for i in range(nb_consommateurs):
    col='conso_'+str(i)
    utilite_choisie[col]=utilite[[col+'_j'+j for j in ['0','1','2']]].max(axis=1)

#utilité de l'eau
utilite_eau=utilite_choisie[['conso_'+str(i)+'_j0' for i in range(nb_consommateurs)]].copy()
for col in utilite_eau.columns:
    utilite_eau.rename(columns={col:col[:-3]},inplace=True)

#utilité moyenne avant fusion
utilite_avant_fusion=utilite_choisie[['conso_'+str(i) for i in range(nb_consommateurs)]].mean().mean()

#boisson gardée en fusion et les paramètres qui correspondent
ville300ter['boisson_fusion']=1+(ville300ter['sucres2']<=ville300ter['sucres1'])
for col in ['sucres','sucres']:
    ville300ter[col+'_fusion']=(
            ville300ter[col+'1']*(ville300ter['boisson_fusion']==1) +
            ville300ter[col+'2']*(ville300ter['boisson_fusion']==2))

for col in ['couts','xi']:
    ville300ter[col+'_fusion']=(
                ville300ter[col+'1_hat']*(ville300ter['boisson_fusion']==1) +
                ville300ter[col+'2_hat']*(ville300ter['boisson_fusion']==2))





def condition(prix,villes): #condition de l'équilibre
    delta=(alpha_hat+beta1_hat*villes['sucres_fusion']-beta2_hat*villes['sucres_fusion']**2-
           gam_hat*prix+villes['xi_fusion'])
    return(1-gam*(prix-villes['couts_fusion'])*(1-delta/(1+delta)))

def prix_equilibre(villes): #résolution numérique de l'équilibre
    result=(scipy.optimize.fsolve(lambda x: condition(x, villes),
                                 villes['couts_fusion']+0.1#,fprime=lambda x: multi_jacobian(x, sucres, xis, couts) 
                                 ))
    result_cleaned = [p if p > 0 else villes['couts_fusion'][n] for n,p in enumerate(result)]
    return(result_cleaned)
    

ville300ter['prix_fusion'] = pd.Series(prix_equilibre(ville300ter))
prix_avant_fusion=ville300ter['prix_1']*(ville300ter['boisson_fusion']==1)+ville300ter['prix_2']*(ville300ter['boisson_fusion']==2)
prix_moyen_avant_fusion=prix_avant_fusion.mean()
prix_moyen_apres_fusion=ville300ter['prix_fusion'].mean()


def calculer_utilite4(villes): #utilité après fusion
    utilite=villes.copy()
    for i in range(nb_consommateurs):
        col='conso_'+str(i)
        utilite[col]=alpha+beta1*utilite['sucres_fusion']-beta2*utilite['sucres_fusion']**2-gam*utilite['prix_fusion']+utilite['xi_fusion']
        utilite[col+'_eau']=utilite_eau[col].copy()
        utilite[col]=utilite[[col,col+'_eau']].max(axis=1)
        del utilite[col+'_eau']
    return(utilite)

def utilite_moyenne(utilite):
    return(utilite.mean().mean())
    
utilite_fusion=utilite_moyenne(calculer_utilite4(ville300ter))

#on cherche la réduction de coût nécessaire pour que le consommateur ne soit pas perdant
ville300quad=ville300ter.copy()
i=1
reduc=0
while utilite_fusion<utilite_avant_fusion and reduc<1:
    reduc=0.05*i 
    ville300quad['couts_fusion']=(1-reduc)*ville300ter['couts_fusion']
    ville300quad['prix_fusion'] = pd.Series(prix_equilibre(ville300quad))
    utilite_fusion=utilite_moyenne(calculer_utilite4(ville300quad))
    i+=1

print(utilite_avant_fusion,reduc,utilite_fusion)
#0.10932707573542158 0.1 0.46518564438499516