#dataset located here: https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data
#The emotions are classified into six categories: 
#sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('emotions.csv')
print(df.head())

#distribution of emotion data among the 4 categories
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Emotion Distribution')
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.show()

#distribution of length of text
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df, x='text_length', bins=30, kde=True, color='skyblue')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

#emotions vs text length
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='text_length', data=df, palette='pastel')
plt.title('Emotion vs. Text Length')
plt.xlabel('Emotion Label')
plt.ylabel('Text Length')
plt.show()

#word cloud for each emotion
from wordcloud import WordCloud

emotions = df['label'].unique()
plt.figure(figsize=(15, 10))
for emotion in emotions:
    subset = df[df['label'] == emotion]
    text = ' '.join(subset['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.subplot(3, 3, emotion+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - Emotion {emotion}')
    plt.axis('off')
plt.show()


#Training set / testing set split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=0)

#Text Vectorization
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#XGBoost Model Training
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=6, random_state=42)
xgb_model.fit(X_train_vec, y_train)

#Model evalution
y_pred = xgb_model.predict(X_test_vec)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred, labels=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])


#Bar chart for classification report
def plot_classification_report(report):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report_dict['accuracy']
    del report_dict['accuracy']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(report_dict['weighted avg'].keys()), y=list(report_dict['weighted avg'].values()), palette='viridis')
    plt.title('Classification Report Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.ylim(0, 1)  
    plt.text(0, 0.95, f'Accuracy: {accuracy:.2f}', ha='left', va='center', color='white', fontsize=12, fontweight='bold')
    plt.show()

plot_classification_report(classification_report(y_test, y_pred))