import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def train_model():
    """
    Train a Random Forest Classifier on the Iris dataset
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the Random Forest Classifier
    # Using more trees than default for better performance
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        max_depth=3,       # Maximum depth of trees
        random_state=42,   # For reproducibility
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    print("Training Random Forest Classifier...")
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Print feature importance (Random Forest advantage)
    print("\nFeature Importance:")
    feature_names = iris.feature_names
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # Create model directory if it doesn't exist
    model_dir = "../model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, "iris_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved successfully at: {model_path}")
    return model

if __name__ == "__main__":
    train_model()
