shots_data=pd.read_csv(rpath+'shots_metadaDf.csv',sep=",",index_col=0)
metadata = pd.read_csv(rpath+'metadata.csv',sep=";")
spk_ftrs=pd.read_csv(rpath+'speaker_ftrs.csv',encode='utf-8',sep=",")
tf_idf = pd.read_csv(rpath+'Transcrption_Tfidf.csv',sep=";",encoding = 'ISO-8859-1')
dfs = (mtdata.merge(attributes, on='filename')).merge(spk_ftrs, on='filename').merge(tags, on='filename').merge(metadata,on='filename')
feats = ['filename','has_text', 'nb_faces_max', 'nb_shots', 'vd_duration', 'size', 'uploader_id','categ','entropy','nb_M','nb_F','Freq_M/F']
df = dfs[feats]
df.set_index('filename',inplace=True)
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

clf = RandomForestClassifier(n_estimators = 50)
y = df['labels']
X = df.drop(['labels'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(
	X, y, test_size=0.25, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
prec = metrics.precision_score(y_test, y_pred, average='micro')
print('precision du classfieur :', prec)
