# %%

import pandas as pd

df = pd.read_parquet("../data/dados_clones.parquet")

df.head()


# %%

from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

features_to_encode = [
      "Distância Ombro a ombro", 
    "Tamanho do crânio", 
    "Tamanho dos pés"
]

features_num = [
    'Massa(em kilos)', 
    'Estatura(cm)', 
    'Tempo de existência(em meses)'
]

# 3. Aplicar o OrdinalEncoder
# handle_unknown='use_encoded_value' ajuda se aparecer uma categoria nova no futuro (opcional)
encoder = OrdinalEncoder()
# O scikit-learn gosta de arrays 2D, então passamos apenas as colunas selecionadas
# É importante sobrescrever apenas as colunas que foram transformadas
df[features_to_encode] = encoder.fit_transform(df[features_to_encode])

# 3. Criar X (Juntando as duas listas)
# O ERRO ESTAVA AQUI: Você deve somar as listas (+) para criar uma lista completa
features = features_to_encode + features_num
X = df[features]

le = LabelEncoder()
y = le.fit_transform(df['Status '])

arvore_clones = tree.DecisionTreeClassifier()

arvore_clones = arvore_clones.fit(X,y)

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400, figsize=[4,4])
tree.plot_tree(arvore_clones,
               feature_names=features,
               class_names=le.classes_,
               filled=True)


# %%
