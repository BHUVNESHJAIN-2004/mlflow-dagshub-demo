import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='bhuvneshjain2004', repo_name='mlflow-dagshub-demo', mlflow=True)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 1
n_estimators=100
# Start an MLflow run

# this is called context manager in python we use it because we doing start run and without it's use we need to  write end_run but by it's use we don't need to write end_run
mlflow.set_tracking_uri("https://dagshub.com/bhuvneshjain2004/mlflow-dagshub-demo.mlflow")
mlflow.set_experiment("iris-rf")
with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train) 
    y_pred = rf.predict(X_test)     
    accuracy = accuracy_score(y_test,y_pred)

    # log metric and params
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png") # first hamari file save ho jayegi above line se aur fir hmne yha path bta diya 

    # log code
    mlflow.log_artifact(__file__)

    # log model
    mlflow.sklearn.log_model(rf,name="Random Forest") # we also write direct .log_model but it is more better
    mlflow.set_tag('author','rahul')
    mlflow.set_tag('model','random forest')
    print('accuracy',accuracy)
    
