# import required libraries
import pandas as pd
import warnings
import pickle
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from itertools import count


# read dataset function
def read_dataset(inFile):
    print("\nReading:", inFile)
    data =  pd.read_json(inFile, lines=True)
    return data

# data paths and config
inTrain = 'subtaskA_train_monolingual.jsonl'
inTest = 'subtaskA_dev_monolingual.jsonl'

max_instances_per_class = 1500
max_features = 20000 # maximum number of features extracted for our instances
random_seed = 777 # set random seed for reproducibility
id2label = {0: "human", 1: "machine"}

# read dataset
train_df = read_dataset(inTrain)
test_df = read_dataset(inTest)

# downsample training data to train faster
train_df = train_df.groupby("label").sample(n=max_instances_per_class, random_state=random_seed)

# vectorize data: extract features from our data (from text to numeric vectors)
vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1,1))
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

# vectorize labels : from text to numeric vectors
le = LabelEncoder()
Y_train = le.fit_transform(train_df["label"])
Y_test = le.transform(test_df["label"])

# create model
model = ExtraTreesClassifier()

# train model
model.fit(X_train, Y_train)

# get test predictions
predictions = model.predict(X_test)

# evaluate predictions
target_names = [label for idx, label in id2label.items()]
print(classification_report(Y_test, predictions, target_names=target_names))

# Pickle
modelA_filename = 'taskA_trained_model.pkl'

with open(modelA_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Modelo guardado como {modelA_filename}")


"""
# classify your own text
custom_texts = ["I'm ChatGPT, your virtual assistant, and I'm generating texts"]
X_custom = vectorizer.transform(custom_texts)
preds = model.predict(X_custom)
print("Classification label:", target_names[preds[0]])

# Filtrar todas las advertencias de convergencia
warnings.filterwarnings('ignore', category=ConvergenceWarning)

best_score = float('-inf')
best_model = None
top_modelos = []

print('Calculando el Mejor Modelo...')
for name, ClassifierClass in all_estimators(type_filter='classifier'):
      if issubclass(ClassifierClass, ClassifierMixin) and hasattr(ClassifierClass, 'fit'):
        try:
            regressor = ClassifierClass()
            regressor.fit(X_train, Y_train)
            y_pred = regressor.predict(X_test)
            score = f1_score(Y_test, y_pred, average="macro")
            top_modelos.append((score, name, regressor))
            if score > best_score:
                best_score = score
                best_model = regressor
            #print(f"Modelo : {name} | Macro F1: {score}")
        except Exception as e:
          print('.')

#Ordenamos los modelos de mayor a menor
top_modelos.sort(reverse=True, key=lambda x: x[0])

#Establecemos el top de mejores modelos
top_five = top_modelos[:5]

#Establecemos el formato para la tabla
print('\n{:_<50}'.format(""))
print("\n{:^50}".format("--- TOP 5 MEJORES MODELOS ---"))
print('{:_<50}'.format(""))
print("\n{:^5} | {:^25} | {:^15}".format("TOP", "MODELO", "PUNTUACIÓN"))
print('{:_<50}\n'.format(""))

#Imprimimos el top 5 modelos
for i, (score, name, model) in enumerate(top_five, start=1):
  recommended = "<- Modelo Recomendado" if model == best_model else ""
  print("{:^5} | {:^25} | {:^13.6f} | {}".format(i, name, score, recommended))
  """