#%% 

import pandas as pd
import mlflow
from sklearn import ensemble, linear_model, metrics, model_selection, tree, pipeline
import matplotlib.pyplot as plt
from feature_engine import discretisation, encoding

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=1)


# %%
#Defining out of time (oot) from the dataset

df = pd.read_csv("../data/abt_churn.csv")
df.head()

oot = df[df['dtRef'] == df['dtRef'].max()].copy()

df_train = df[df['dtRef'] < df['dtRef'].max()].copy()

# %%
features = df_train.columns[2:-1]
target = 'flagChurn'

X,y = df_train[features], df_train[target]

# %%
# SAMPLE

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify = y
                                                                    )                                                                    

# Verificar a representatividade da variavel resposta entre treino e teste 
# train_test_split statify permite balancear a base de dados

print('Taxa variável resposta Treino:',y_train.mean())
print('Taxa variável resposta Teste:',y_test.mean())
print('Tava variável reposta geral:',y.mean()) 
# Importante verificar se o evento é raro: taxa de acontecimento <5% ou >95%


# %%
# Exploratory Data Analysis
#Explore (missings)

X_train.isna().sum().sort_values(ascending=False)

# Análise bivariada

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean','median']).T
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'],ascending=False)

# Descobrir melhores features

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train,y_train)
feature_importances = (pd.Series(arvore.feature_importances_,index=X_train.columns)
                       .sort_values(ascending=False)
                       .reset_index()
                       )

feature_importances['acum'] = feature_importances[0].cumsum()
best_features = (feature_importances[feature_importances['acum'] < 0.96]['index']
                 .tolist())


# %%
#MODIFY

#Data discretization
tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                              regression = False,
                                                              bin_output='bin_number',
                                                              cv=3)

# Onehot enconding
onehot = encoding.OneHotEncoder(variables = best_features,ignore_format=True)

# %%
# MODEL
# model = linear_model.LogisticRegression(random_state=42,
#                                         max_iter=1000,)
model = ensemble.RandomForestClassifier(
    random_state=42,
    n_jobs=2,
 )


params = {
    "min_samples_leaf":[15,20,25,30,50],
    "n_estimators":[100,200,500,1000],
    "criterion":['gini', 'entropy', 'log_loss']
}

# params = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'saga'],
#     'class_weight': [None, 'balanced']
# }

grid = model_selection.GridSearchCV(model, 
                                    params, 
                                    cv=3,
                                    scoring='roc_auc',
                                    verbose=4)


model_pipeline = pipeline.Pipeline(
    steps= [
        ('Discretizar',tree_discretization),
        ('Onehot', onehot),
        ('Grid',grid)
    ]
)


with mlflow.start_run(run_name=model.__str__()):
    mlflow.autolog()
    model_pipeline.fit(X_train,y_train)

#ASSESS
    y_train_predict = model_pipeline.predict(X_train)
    y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

    acc_train = metrics.accuracy_score(y_train,y_train_predict)
    auc_train = metrics.roc_auc_score(y_train,y_train_proba)
    roc_train = metrics.roc_curve(y_train,y_train_proba)

    print("Acurácia Treino:",acc_train)
    print("AUC Treino:",auc_train)

    y_test_predict = model_pipeline.predict(X_test)
    y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

    acc_test = metrics.accuracy_score(y_test,y_test_predict)
    auc_test = metrics.roc_auc_score(y_test,y_test_proba)
    roc_test = metrics.roc_curve(y_test,y_test_proba)
    print("Acurácia Teste:",acc_test)
    print("AUC Teste:",auc_test)

    y_oot_predict = model_pipeline.predict(oot[features])
    y_oot_proba = model_pipeline.predict_proba(oot[features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target],y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target],y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target],y_oot_proba)
    print("Acurácia oot:",acc_oot)
    print("AUC oot:",auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot
        })

# %%

# Extract the name of the model from the pipeline
model_name = model_pipeline.steps[-1][1].__class__.__name__

plt.plot(roc_train[0],roc_train[1])
plt.plot(roc_test[0],roc_test[1])
plt.plot(roc_oot[0],roc_oot[1])
plt.plot([0,1],[0,1],'--',color='black')
plt.grid(True)
plt.title(f'ROC Curve - {model_name}')
plt.legend(
    [f"Treino: {100*auc_train:.2f}",
     f"Teste: {100*auc_test:.2f}",
     f"Out of Time: {100*auc_oot:.2f}",
    ]
)
plt.show()
# %%

model_df = pd.Series({
    "model":model_pipeline,
    "features":best_features,
    })

model_df.to_pickle("model.pkl")