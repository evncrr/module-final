import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#load data
name_df = pd.read_csv("mlb/Master.csv")[['playerID', 'nameFirst', 'nameLast']]
hof_df = pd.read_csv("mlb/HallOfFame.csv")[['playerID', 'inducted']]
batting_df = pd.read_csv("mlb/Batting.csv")
allstar_df = pd.read_csv("mlb/AllstarFull.csv")
awards_df = pd.read_csv("mlb/AwardsPlayers.csv")

#player name
name_df['player_Name'] = name_df['nameFirst'] + " " + name_df['nameLast']

#encode HoF
hof_df['inducted'] = hof_df['inducted'].apply(lambda x: 1 if x == "Y" else 0)
hof_df = hof_df.groupby('playerID')['inducted'].max().reset_index()

#add stats
career_stats = batting_df.groupby('playerID')[['G', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SB', 'RBI']].sum().reset_index()
allstar_counts = allstar_df.groupby('playerID').size().reset_index(name='Allstar_Appearances')
awards_counts = awards_df.groupby('playerID').size().reset_index(name='Awards')

#merge data
df = career_stats.merge(allstar_counts, on='playerID', how='left').fillna(0)
df = df.merge(awards_counts, on='playerID', how='left').fillna(0)
df = df.merge(name_df[['playerID', 'player_Name']], on='playerID', how='left')
df = df.merge(hof_df, on='playerID', how='left').fillna(0)

#filter data
df = df[df['AB'] >= 1000].reset_index(drop=True)

#add averages
df['BA'] = df['H'] / df['AB']
df['OBP'] = (df['H'] + df['BB']) / (df['AB'] + df['BB'])
df['SLG'] = (df['H'] - df['2B'] - df['3B'] - df['HR'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / df['AB']
df['OPS'] = df['OBP'] + df['SLG']

#features and labels
features = ['G', 'AB', 'BA', 'OPS', 'H', '2B', '3B', 'HR', 'BB', 'SB', 'RBI', 'Allstar_Appearances', 'Awards']
X = df[features]
y = df['inducted']

#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#XGBoost Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=25, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("\nXGBoost Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

#clustering
df['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)

#PCA
pca_result = PCA(n_components=2).fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', style='inducted', palette='Set2')
plt.title('PCA Clusters of MLB Players and HOF Induction')
plt.show()

#heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features + ['inducted']].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

#similarity matrix
dist_matrix = euclidean_distances(X_scaled)
similarity_matrix = 1 / (1 + dist_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=df['player_Name'], columns=df['player_Name'])

target_player = "Barry Bonds"
if target_player in similarity_df.columns:
    similar_players = similarity_df[target_player].sort_values(ascending=False).head()
    print(f"\nPlayers most similar to {target_player}:")
    print(similar_players)

#cluster summary
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster Feature Means:")
print(cluster_summary)

#view sample players in each cluster
for cluster_num in range(4):
    print(f"\nPlayers in Cluster {cluster_num}:")
    cluster_players = df[df['Cluster'] == cluster_num]
    print(cluster_players[['player_Name', 'HR', 'RBI', 'BA', 'Allstar_Appearances', 'Awards']].head())
