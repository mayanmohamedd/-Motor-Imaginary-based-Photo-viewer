
import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def read_data(path):
  raw=mne.io.read_raw_gdf(path,preload=True,
                          eog=['EOG-left', 'EOG-central', 'EOG-right'])
  raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
  raw.set_eeg_reference()
  events=mne.events_from_annotations(raw)
  epochs = mne.Epochs(raw, events[0], event_id=[7,8],on_missing ='warn')
  labels=epochs.events[:,-1]
  features=epochs.get_data()
  return features,labels

features,labels,groups=[],[],[]
for i in range(1,10):
  feature,label=read_data(f'archive (3)/A0{i}T.gdf')
  features.append(feature)
  labels.append(label)
  groups.append([i]*len(label))

features=np.concatenate(features)
labels=np.concatenate(labels)
groups=np.concatenate(groups)

print(features.shape,labels.shape,groups.shape)

print("number of null values in features",np.isnan(features).sum())

unique, counts = np.unique(labels, return_counts=True)
print("unique values in label are",unique, "unique counts in label are",counts)

labels = np.where(labels == 7, 0, 1)

unique2, counts2 = np.unique(groups, return_counts=True)
print("unique values in groups are",unique2, "unique counts in groups are",counts2)


from mne.decoding import CSP
from scipy.signal import butter, filtfilt

def apply_car(signal):
    avg = np.mean(signal, axis=0)
    return signal - avg

def apply_z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

def apply_butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    print("Applied bandpass filter with interval:", (lowcut, highcut))
    return filtered_data


# Initialize lists to store CSP transformed features
csp_transformed_features = []

# Choose one group (subject) from the dataset
group_idx = 1 # Choose the index of the group you want to process
# Extract features and labels for the selected group
group_features = features[groups == group_idx]
group_labels = labels[groups == group_idx]
group_features_car = apply_car(group_features)

# Loop through the frequency range with the specified interval step
#for lowcut in range(4, highcut, interval_step):
 #   highcut = lowcut + interval_step
    
    # Apply bandpass filter
filtered_features = apply_butter_bandpass_filter(group_features_car, 8, 30, fs=250)
print("Shape of filtered features before CSP:", filtered_features.shape)

# Apply z-score normalization
normalized_features = apply_z_score_normalization(filtered_features)
print("Shape of normalized features:", normalized_features.shape)
    
    # Apply CSP
csp = CSP(n_components=14)
csp.fit(normalized_features, group_labels)
    
    # Transform features using CSP
transformed_features = csp.transform(normalized_features)
print("Shape of transformed features after CSP:", transformed_features.shape)

    # Append transformed features to the list
csp_transformed_features.append(transformed_features)

# Concatenate transformed features from all frequency intervals
csp_transformed_features = np.concatenate(csp_transformed_features, axis=-1)

# Check the shape of transformed features
print("Shape of CSP transformed features for group", group_idx, ":", csp_transformed_features.shape)

X_train, X_test, y_train, y_test = train_test_split(csp_transformed_features, group_labels, test_size=0.2, random_state=42)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
scaler2=StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc2 = scaler.fit_transform(X_train)
X_test_sc2 = scaler.transform(X_test)





def train_and_evaluate(clf, params, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(clf, params, cv=5)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    train_accuracy = best_clf.score(X_train, y_train)
    test_accuracy = best_clf.score(X_test, y_test)
    
    return best_clf, train_accuracy, test_accuracy

# Define parameters to search for Logistic Regression
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Train and evaluate Logistic Regression classifier
best_lr_clf, lr_train_accuracy, lr_test_accuracy = train_and_evaluate(clf=LogisticRegression(), 
                                                                      params=lr_params, 
                                                                      X_train=X_train_sc, 
                                                                      y_train=y_train, 
                                                                      X_test=X_test_sc, 
                                                                      y_test=y_test)

print("Logistic Regression Training Accuracy:", lr_train_accuracy)
print("Logistic Regression Testing Accuracy:", lr_test_accuracy)

# Define parameters to search for SVM
svm_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 
              'kernel': ['linear', 'rbf']}

# Train and evaluate SVM classifier
best_svm_clf, svm_train_accuracy, svm_test_accuracy = train_and_evaluate(clf=SVC(), 
                                                                         params=svm_params, 
                                                                         X_train=X_train_sc, 
                                                                         y_train=y_train, 
                                                                         X_test=X_test_sc, 
                                                                         y_test=y_test)

print("SVM Training Accuracy:", svm_train_accuracy)
print("SVM Testing Accuracy:", svm_test_accuracy)

# Define parameters to search for Random Forest
rf_params = {'n_estimators': [100, 150, 200],
             'max_depth': [None, 4],
             'random_state': [123]}

# Train and evaluate Random Forest classifier
best_rf_clf, rf_train_accuracy, rf_test_accuracy = train_and_evaluate(clf=RandomForestClassifier(), 
                                                                      params=rf_params, 
                                                                      X_train=X_train_sc, 
                                                                      y_train=y_train, 
                                                                      X_test=X_test_sc, 
                                                                      y_test=y_test)

print("Random Forest Training Accuracy1:", rf_train_accuracy)
print("Random Forest Testing Accuracy1:", rf_test_accuracy)


# Define parameters for multinomial Naive Bayes
nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
# Scale the features

# Initialize multinomial Naive Bayes classifier
nb_clf = MultinomialNB()

# Train and evaluate multinomial Naive Bayes classifier
best_nb_clf, nb_train_accuracy, nb_test_accuracy = train_and_evaluate(nb_clf, nb_params, X_train_sc, y_train, X_test_sc, y_test)

# Print the results
print("Multinomial Naive Bayes Training Accuracy:", nb_train_accuracy)
print("Multinomial Naive Bayes Testing Accuracy:", nb_test_accuracy)


# Define parameters for Gaussian Naive Bayes
gnb_clf = GaussianNB()

# Train and evaluate Gaussian Naive Bayes classifier
best_gnb_clf, gnb_train_accuracy, gnb_test_accuracy = train_and_evaluate(gnb_clf, {}, X_train_sc, y_train, X_test_sc, y_test)

# Print the results
print("Gaussian Naive Bayes Training Accuracy:", gnb_train_accuracy)
print("Gaussian Naive Bayes Testing Accuracy:", gnb_test_accuracy)




from sklearn.naive_bayes import ComplementNB

# Define parameters for Complement Naive Bayes
cnb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}

# Initialize Complement Naive Bayes classifier
cnb_clf = ComplementNB()

# Train and evaluate Complement Naive Bayes classifier
best_cnb_clf, cnb_train_accuracy, cnb_test_accuracy = train_and_evaluate(cnb_clf, cnb_params, X_train_sc, y_train, X_test_sc, y_test)

# Print the results
print("Complement Naive Bayes Training Accuracy:", cnb_train_accuracy)
print("Complement Naive Bayes Testing Accuracy:", cnb_test_accuracy)






# Compare the accuracies of all classifiers
accuracies = {
    'Logistic Regression': lr_test_accuracy,
    'SVM': svm_test_accuracy,
    'Random Forest': rf_test_accuracy,
    'Multinomial Naive Bayes':nb_test_accuracy,
    'Gaussian Naive Bayes':gnb_test_accuracy,
    'Complement Naive Bayes':cnb_test_accuracy,

}

best_classifier = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_classifier]


print(f"\nThe best classifier is {best_classifier} with an accuracy of {best_accuracy:.2f}.")



