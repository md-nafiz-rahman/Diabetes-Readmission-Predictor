#!/usr/bin/env python
# coding: utf-8

# # PART 1

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# In[2]:


def load_data(file_path):
    # load dataset from CSV file.
    return pd.read_csv(file_path)


# In[3]:


def preprocess_data(df):
    # Replace missing values marked with '?' with np.nan
    df.replace('?', np.nan, inplace=True)
   
    # Set the threshold for dropping columns with missing values
    threshold_drop_missing = 0.5  # 50% missing values
    # Drop columns with more than 50% missing values
    df = df.dropna(thresh=len(df) * (1 - threshold_drop_missing), axis=1)
   
    # Drop columns where over 95% of the values are the same
    threshold_unique = 0.95
    for col in df.columns:
        if df[col].value_counts(normalize=True).iloc[0] > threshold_unique:
            df = df.drop(col, axis=1)
               
    # Apply age_to_midpoint function to 'age' column
    df['age'] = df['age'].apply(age_to_midpoint)
   
    # Fill missing values in 'diag_1', 'diag_2', and 'diag_3' columns with 0
    df[['diag_1', 'diag_2', 'diag_3']] = df[['diag_1', 'diag_2', 'diag_3']].fillna(0)
   
    # Remove rows with missing values
    df.dropna(inplace=True)
   
    # Calculate z-scores for numerical columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    z_scores = stats.zscore(df[numerical_features])
   
    # Identify outliers and remove them
    outliers = (np.abs(z_scores) > 3).any(axis=1)
    df = df[~outliers]
   
    # Drop duplicate rows based on 'patient_nbr'
    df.drop_duplicates(subset='patient_nbr', inplace=True)
   
    return df


# In[4]:


def age_to_midpoint(age_range):
    # Convert age ranges to midpoint values.
    age_ranges = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
                  '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
    return age_ranges.get(age_range)


# In[5]:


def check_distinct_values(df, categorical_columns):
    # Check distinct values and their frequencies for categorical columns.
    for col in categorical_columns:
        print(f"Column: {col}")
        distinct_values = df[col].value_counts()
        print(f"Number of distinct values: {len(distinct_values)}")
        print("Distinct Values and their Frequencies:")
        print(distinct_values)
        print("------------------------------")


# In[6]:


# Function to map the ICD codes from diag columns to their categories
def map_icd_code_to_category(code):
    try:
        code = float(code)
        if 0 <= code < 1:
            return 'Infectious and Parasitic Diseases'
        elif 1 <= code <= 139:
            return 'Infectious and Parasitic Diseases'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        elif 240 <= code <= 249 or 251 <= code <= 279:
            return 'Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders'
        elif 250.00 <= code <= 250.99:
            return 'Diabetes mellitus'
        elif 280 <= code <= 289:
            return 'Diseases of the Blood and Blood-forming Organs'
        elif 290 <= code <= 319:
            return 'Mental Disorders'
        elif 320 <= code <= 359:
            return 'Diseases of the Nervous System'
        elif 360 <= code <= 389:
            return 'Diseases of the sense organs'
        elif 390 <= code <= 459 or code == 785:
            return 'Disease of the Circulatory System'
        elif 460 <= code <= 519 or code == 786:
            return 'Disease of the Respiratory System'
        elif 520 <= code <= 579 or code == 787:
            return 'Diseases of the Digestive System'
        elif 580 <= code <= 629 or code == 788:
            return 'Diseases of the Genitourinary System'
        elif 630 <= code <= 679:
            return 'Complications of Pregnancy, Childbirth, and the Puerperium'
        elif 680 <= code <= 709 or code == 782:
            return 'Diseases of the Skin and Subcutaneous Tissue'
        elif 710 <= code <= 739:
            return 'Diseases of the Musculoskeletal System and Connective Tissue'
        elif 740 <= code <= 759:
            return 'Congenital Anomalies'
        elif 760 <= code <= 779:
            return 'Certain Conditions originating in the Perinatal Period'
        elif 790 <= code <= 799 or code == 780 or code == 781 or code == 784:
            return 'Other symptoms, signs, and ill-defined conditions'
        elif 800 <= code <= 999:
            return 'Injury and Poisoning'
        else:
            return 'Other'
    except ValueError:
        if 'E' in code or 'V' in code:
            return 'External causes of injury and supplemental classification'
        else:
            return 'Other'


# In[7]:


def visualise_categorical_data(df, column):
    # Visualise categorical data with percentages.
    plt.figure(figsize=(14,14))
    
    # Creating the countplot
    ax = sns.countplot(x=column, hue='readmitted', data=df)
    plt.title(f'{column.capitalize()} vs. Readmission')
    plt.xlabel(column.capitalize())
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.legend(title='Readmitted', loc='upper right')
    
    # Calculating percentages
    total = len(df[column])
    for p in ax.patches:
        height = p.get_height()
        # Calculating and annotating percentage for each bar
        ax.text(p.get_x() + p.get_width() / 2., height + 4, '{:1.2f}%'.format((height / total) * 100), ha="center")
    
    plt.tight_layout()
    plt.show()


# In[8]:


# Load the diabetic_data.csv file
df = load_data('diabetic_data.csv')

print('Initial dataframe shape:')
print(df.shape)


# In[9]:


# Preprocess the dataset
df = preprocess_data(df)


# In[10]:


# Distinct values and the frequencies for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
check_distinct_values(df, categorical_columns)


# In[11]:


print('Final dataframe shape:')
print(df.shape)


# In[12]:


# Replace 'NO' with 0, '<30' and '>30' with 1 in the 'readmitted' column
replacement_map = {'NO': 0, '<30': 1, '>30': 1}
unique_values = df['readmitted'].unique()
print(unique_values)
df['readmitted'] = df['readmitted'].map(replacement_map)

# Display the unique values in the 'readmitted' column
unique_values = df['readmitted'].unique()
print(unique_values)


# In[13]:


# Read the icd_codes.csv file
icd_codes = pd.read_csv('icd_codes.csv')

# Join datasets on 'diag_1' column
df = pd.merge(df, icd_codes, how='left', left_on='diag_1', right_on='ICD_Code')

# Apply map_icd_code_to_category function to 'diag_1', 'diag_2', and 'diag_3' columns
df['diag_1'] = df['diag_1'].apply(map_icd_code_to_category)
df['diag_2'] = df['diag_2'].apply(map_icd_code_to_category)
df['diag_3'] = df['diag_3'].apply(map_icd_code_to_category)


# In[14]:


# Visualisations
visualise_categorical_data(df, 'age')

print(df['age'].value_counts())

visualise_categorical_data(df, 'race')

print(df['race'].value_counts())

visualise_categorical_data(df, 'gender')

print(df['gender'].value_counts())


# In[15]:


# Plotting for diag_1
visualise_categorical_data(df, 'diag_1')

print(df['diag_1'].value_counts())

# Plotting for diag_2
visualise_categorical_data(df, 'diag_2')

print(df['diag_2'].value_counts())

# Plotting for diag_3
visualise_categorical_data(df, 'diag_3')

print(df['diag_3'].value_counts())


# ## Model Building

# In[16]:


# Selecting the required features for model building
selected_columns = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital',
                    'number_inpatient', 'encounter_id', 'age', 'num_lab_procedures',
                    'number_diagnoses', 'num_procedures', 'readmitted']

# Creating features (X) and target variable (y)
X = df[selected_columns].drop(columns=['readmitted'])
y = df['readmitted']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Building the Random Forest Classifier model
model = RandomForestClassifier(oob_score=True, random_state=0)
model.fit(X_train, y_train)

# Prediction on the test set
y_pred = model.predict(X_test)

# Calculating and plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluating the model using additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Evaluating the model using cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
# Print the OOB score
print(f'RandomForest - OOB Score: {model.oob_score_:.4f}')
# Printing the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# # IMPROVED MODEL PART 2

# ## Exploratory Analysis Part 2

# In[17]:


# Import the packages 
import IPython
import scipy as sp
from statistics import mode
from scipy.stats import norm


# In[18]:


# Read the dataset
data = pd.read_csv("diabetic_data.csv")

# Check the size
data.shape


# In[19]:


# Check the dataset data types and null values.
data.info()


# In[20]:


# NO null values

# Check the data types only for object types.
for i in data.columns:
    if data[i].dtype==object:
        d= data[i].value_counts()
        print(pd.DataFrame(data=d))


# In[21]:


# There are no null values in the dataset.

# Check the numerical data values 
data.describe().transpose()


# In[22]:


# The data shows that the longest hospital stay recorded is 14 days.
# 'encounter_id' and 'patient_nbr' need to be treated or dropped. 
# On average, the number of lab procedures is 43. 
# Additionally, patients are typically prescribed an average of 16 medications
# The mean of diagnoses made is approximately 7.4.


# In[23]:


# Count the occurrences of each category in the 'readmitted' column.
target_count=data['readmitted'].value_counts()


# In[24]:


# Plot the count of each category as a bar plot
target_count.plot(kind='bar', title='Readmission_count')


# In[25]:


# Figure and axes for the plots
fig = plt.figure(figsize=(16, 10))

# Count occurrences for age and gender
age_count = data['age'].value_counts()
gender_count = data['gender'].value_counts()

# Subplot for age distribution
ax_age = fig.add_subplot(2, 2, 1)
age_count.plot(kind='bar', title='Age Distribution', ax=ax_age)

# Subplot for gender distribution
ax_gender = fig.add_subplot(2, 2, 2)
gender_count.plot(kind='bar', title='Gender Distribution', ax=ax_gender)

# Count occurrences for race and weight
race_count = data['race'].value_counts()
weight_count = data['weight'].value_counts()

# Subplot for race distribution
ax_race = fig.add_subplot(2, 2, 3)
race_count.plot(kind='bar', title='Race Distribution', ax=ax_race)

# Subplot for weight distribution
ax_weight = fig.add_subplot(2, 2, 4)
weight_count.plot(kind='bar', title='Weight Distribution', ax=ax_weight)

plt.tight_layout()


# In[26]:


# Retrieve numerical columns excluding the 'readmitted' column.
num_col = list(set(list(data._get_numeric_data().columns))- {'readmitted'})


# In[27]:


sns.set()

# Plot the numerical features
sns.pairplot(data[num_col], height = 2.5)
plt.show()


# ## Enhanced Data Preprocessing

# In[28]:


# Copy of the file for pre-processing 
train = data.copy(deep=True)

# Initialize an empty list to store the results.
missing_data = []


# In[29]:


# Identify object columns
object_cols = train.select_dtypes(include=['object']).columns
# Check the percentage of missing values for each column.
for col in object_cols:
    count_missing = train[col][train[col] == '?'].count()
    percent_missing = (count_missing / train.shape[0] * 100).round(2)
    missing_data.append([col, count_missing, percent_missing])


# In[30]:


# Create a DataFrame from the list and sort it by the percentage of missing values in descending order.
missing_value = pd.DataFrame(missing_data, columns=["col", "count_missing", "percent_missing"]).sort_values(by="percent_missing", ascending=False)
missing_value


# In[31]:


# Drop the irrelavant and high missing value variables
train = train.drop(columns=['weight', 'medical_specialty', 'payer_code','encounter_id','citoglipton','examide'],axis=1)
# Drop only the missing values if missing in all three diagonosis categories 
train = train.drop(set(train[(train['diag_1']== '?') & (train['diag_2'] == '?') & (train['diag_3'] == '?')].index))
# Drop the expired patients
train = train.drop(train[train['discharge_disposition_id'].isin([11, 19, 20, 21])].index)
# Drop duplicated patients
train = train.drop_duplicates(subset='patient_nbr')
# Irrelevant variable
train=train.drop(['patient_nbr'],axis=1)

train.shape


# In[32]:


print(train['gender'].value_counts())


# In[33]:


#Replacing missing values
train = train.drop(set(train['gender'][train['gender'] == 'Unknown/Invalid'].index))

# Replacing missing values in race with Other category
train['race'] = train['race'].replace('?', 'Other')

train.shape


# In[34]:


# Adding 'patient_service' to track how much hospital care patients receive overall.
# Creating 'med_change' to measure changes in patients medication doses over time.
# Introducing 'num_med' to count how many different medications each patient is taking.

# 'patient_service' sums the counts of outpatient visits, emergency visits, and inpatient admissions.
train['patient_service'] = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']


# In[35]:


# List of medications
keys = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", 
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
    "insulin", "glyburide-metformin", "glipizide-metformin", 
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

# Transform medication to binary and sum up
# Initialize 'med_change' to zero before looping
train['med_change'] = 0

# Loop over each medication convert and sum
for medication in keys:
    new_col_name = medication + 'new'
    train[new_col_name] = train[medication].map(lambda x: 0 if x in ['No', 'Steady'] else 1)
    train['med_change'] += train[new_col_name]

# Remove temporary columns
train.drop(columns=[key + 'new' for key in keys], inplace=True)

# Display the 'med_change' count of each value
train['med_change'].value_counts(dropna=False)


# In[36]:


# Calculating the number of medications used for each patient
# Replace categorical values
med_map = {'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1}

for col in keys:
    train[col] = train[col].map(med_map)

#Initialize 'num_med' column to count medications per patient
train['num_med'] = 0

# Increment 'num_med' by the medication value for each patient
for col in keys:
    train['num_med'] = train['num_med'] + train[col]
    
# Check the distribution of 'num_med' for the occurrences of each value.
train['num_med'].value_counts(dropna=False)


# In[37]:


# diag_1 now primary diagnosis, diag_2 now secondary diagnosis, and diag_3 now additional diagnosis.

# Duplicating the diagnosis columns
train['primary_diag'] = train['diag_1']
train['secondary_diag'] = train['diag_2']
train['additional_diag'] = train['diag_3']


# In[38]:


# Replacing the "?" with -1 in diagnosis columns
diagnosis_columns = ['primary_diag', 'secondary_diag', 'additional_diag']
for column in diagnosis_columns:
    train[column] = train[column].replace('?', -1)


# In[39]:


# Recoding ICD codes for V or E to the “other” category = 0
def recode_ve_to_zero(value):
    if isinstance(value, str) and (value.startswith('V') or value.startswith('E')):
        return 0
    else:
        return value

# Apply the function to each diagnosis
for column in diagnosis_columns:
    train[column] = train[column].apply(recode_ve_to_zero)

# convert the data type to float for later computation
for column in diagnosis_columns:
    train[column] = train[column].astype(float)


# In[40]:


# recoding ICD codes diag1,diag2,diag3 to categories
for index, row in train.iterrows():
    if (row['primary_diag'] >= 390 and row['primary_diag'] < 460) or (np.floor(row['primary_diag']) == 785):
        train.loc[index, 'primary_diag'] = 1
    elif (row['primary_diag'] >= 460 and row['primary_diag'] < 520) or (np.floor(row['primary_diag']) == 786):
        train.loc[index, 'primary_diag'] = 2
    elif (row['primary_diag'] >= 520 and row['primary_diag'] < 580) or (np.floor(row['primary_diag']) == 787):
        train.loc[index, 'primary_diag'] = 3
    elif (np.floor(row['primary_diag']) == 250):
        train.loc[index, 'primary_diag'] = 4
    elif (row['primary_diag'] >= 800 and row['primary_diag'] < 1000):
        train.loc[index, 'primary_diag'] = 5
    elif (row['primary_diag'] >= 710 and row['primary_diag'] < 740):
        train.loc[index, 'primary_diag'] = 6
    elif (row['primary_diag'] >= 580 and row['primary_diag'] < 630) or (np.floor(row['primary_diag']) == 788):
        train.loc[index, 'primary_diag'] = 7
    elif (row['primary_diag'] >= 140 and row['primary_diag'] < 240):
        train.loc[index, 'primary_diag'] = 8
    else:
        train.loc[index, 'primary_diag'] = 0

for index, row in train.iterrows():
    if (row['secondary_diag'] >= 390 and row['secondary_diag'] < 460) or (np.floor(row['secondary_diag']) == 785):
        train.loc[index, 'secondary_diag'] = 1
    elif (row['secondary_diag'] >= 460 and row['secondary_diag'] < 520) or (np.floor(row['secondary_diag']) == 786):
        train.loc[index, 'secondary_diag'] = 2
    elif (row['secondary_diag'] >= 520 and row['secondary_diag'] < 580) or (np.floor(row['secondary_diag']) == 787):
        train.loc[index, 'secondary_diag'] = 3
    elif (np.floor(row['secondary_diag']) == 250):
        train.loc[index, 'secondary_diag'] = 4
    elif (row['secondary_diag'] >= 800 and row['secondary_diag'] < 1000):
        train.loc[index, 'secondary_diag'] = 5
    elif (row['secondary_diag'] >= 710 and row['secondary_diag'] < 740):
        train.loc[index, 'secondary_diag'] = 6
    elif (row['secondary_diag'] >= 580 and row['secondary_diag'] < 630) or (np.floor(row['secondary_diag']) == 788):
        train.loc[index, 'secondary_diag'] = 7
    elif (row['secondary_diag'] >= 140 and row['secondary_diag'] < 240):
        train.loc[index, 'secondary_diag'] = 8
    else:
        train.loc[index, 'secondary_diag'] = 0

for index, row in train.iterrows():
    if (row['additional_diag'] >= 390 and row['additional_diag'] < 460) or (np.floor(row['additional_diag']) == 785):
        train.loc[index, 'additional_diag'] = 1
    elif (row['additional_diag'] >= 460 and row['additional_diag'] < 520) or (np.floor(row['additional_diag']) == 786):
        train.loc[index, 'additional_diag'] = 2
    elif (row['additional_diag'] >= 520 and row['additional_diag'] < 580) or (np.floor(row['additional_diag']) == 787):
        train.loc[index, 'additional_diag'] = 3
    elif (np.floor(row['additional_diag']) == 250):
        train.loc[index, 'additional_diag'] = 4
    elif (row['additional_diag'] >= 800 and row['additional_diag'] < 1000):
        train.loc[index, 'additional_diag'] = 5
    elif (row['additional_diag'] >= 710 and row['additional_diag'] < 740):
        train.loc[index, 'additional_diag'] = 6
    elif (row['additional_diag'] >= 580 and row['additional_diag'] < 630) or (np.floor(row['additional_diag']) == 788):
        train.loc[index, 'additional_diag'] = 7
    elif (row['additional_diag'] >= 140 and row['additional_diag'] < 240):
        train.loc[index, 'additional_diag'] = 8
    else:
        train.loc[index, 'additional_diag'] = 0


# In[41]:


# Display the difference of diagnosis columuns 1/2/3
train[['diag_1', 'primary_diag']].head(15).T


# In[42]:


train[['diag_2','secondary_diag']].head(15).T


# In[43]:


train[['diag_3','additional_diag']].head(15).T


# In[44]:


# Show the distribution of 'admission_type_id' column.
train['admission_type_id'].value_counts(dropna=False)


# In[45]:


# Recategorising admission type id for narrowing
train['admission_type_id'] = train['admission_type_id'].replace(2,1)
train['admission_type_id'] = train['admission_type_id'].replace(7,1)
train['admission_type_id'] = train['admission_type_id'].replace(6,5)
train['admission_type_id'] = train['admission_type_id'].replace(8,5)


# In[46]:


# Analysing the distribution in 'discharge_disposition_id'.
train['discharge_disposition_id'].value_counts(dropna=False)


# In[47]:


# Recategorising discharge_disposition_id for narrowing
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(6,1)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(8,1)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(9,1)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(13,1)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(3,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(4,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(5,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(14,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(22,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(23,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(24,2)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(12,10)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(15,10)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(16,10)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(17,10)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(25,18)
train['discharge_disposition_id'] = train['discharge_disposition_id'].replace(26,18)


# In[48]:


# Analysing the distribution in 'admission_source_id'.
train['admission_source_id'].value_counts(dropna=False)


# In[49]:


# Recategorising admission_source_id for narrowing
train['admission_source_id'] = train['admission_source_id'].replace(2,1)
train['admission_source_id'] = train['admission_source_id'].replace(3,1)
train['admission_source_id'] = train['admission_source_id'].replace(5,4)
train['admission_source_id'] = train['admission_source_id'].replace(6,4)
train['admission_source_id'] = train['admission_source_id'].replace(10,4)
train['admission_source_id'] = train['admission_source_id'].replace(22,4)
train['admission_source_id'] = train['admission_source_id'].replace(25,4)
train['admission_source_id'] = train['admission_source_id'].replace(15,9)
train['admission_source_id'] = train['admission_source_id'].replace(17,9)
train['admission_source_id'] = train['admission_source_id'].replace(20,9)
train['admission_source_id'] = train['admission_source_id'].replace(21,9)
train['admission_source_id'] = train['admission_source_id'].replace(13,11)
train['admission_source_id'] = train['admission_source_id'].replace(14,11)


# In[50]:


# mapping for each column
change_mapping = {'Ch': 1, 'No': 0}
gender_mapping = {'Male': 1, 'Female': 0}
diabetesMed_mapping = {'Yes': 1, 'No': 0}
readmitted_mapping = {'>30': 1, '<30': 1, 'NO': 0}

# Apply the mapping to recode each column
train['change'] = train['change'].map(change_mapping)
train['gender'] = train['gender'].map(gender_mapping)
train['diabetesMed'] = train['diabetesMed'].map(diabetesMed_mapping)
train['readmitted'] = train['readmitted'].map(readmitted_mapping)

print(train[['change', 'gender', 'diabetesMed', 'readmitted']].value_counts(dropna=False))


# In[51]:


# Check the distribution of 'race'
train['race'].value_counts(dropna=False)


# In[52]:


# Remapping the race column
race_mapping = {
    'Caucasian': 1,
    'AfricanAmerican': 2,
    'Hispanic': 3,
    'Asian': 4,
    'Other': 0
}

train['race'] = train['race'].map(race_mapping)
print(train['race'].value_counts(dropna=False))


# In[53]:


# Check the distribution of age
train['age'].value_counts(dropna=False)


# In[54]:


# Recode age groups using the mean of each interval
age_mapping = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}
train['age'] = train['age'].map(age_mapping)
train['age'] = train['age'].astype('int64')
train['age'].value_counts(dropna=False)


# In[55]:


train['max_glu_serum'].value_counts(dropna=False)


# In[56]:


# Mapping max_glu_serum
# 'None' identified as NaN, transform back
train['max_glu_serum'] = train['max_glu_serum'].fillna('None')

max_glu_serum_mapping = {
    '>200': 1,
    '>300': 1,
    'Norm': 0,
    'None': 99
}

train['max_glu_serum'] = train['max_glu_serum'].map(max_glu_serum_mapping)
train['max_glu_serum'].value_counts(dropna=False)


# In[57]:


train['A1Cresult'].value_counts(dropna=False)


# In[58]:


# Mapping A1Cresult
train['A1Cresult'] = train['A1Cresult'].fillna('None')

# 'None' identified as NaN, transform back
A1Cresult_mapping = {
    '>7': 1,
    '>8': 1,
    'Norm': 0,
    'None': 99
}

train['A1Cresult'] = train['A1Cresult'].map(A1Cresult_mapping)
train['A1Cresult'].value_counts(dropna=False)


# ## Feature Engineering 

# In[59]:


train.head(15).T


# In[60]:


# Check the data types for categorical data
train.dtypes


# In[61]:


# Change categorical columns to 'object' type to optimize processing
i = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
     'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
     'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
     'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
     'max_glu_serum', 'primary_diag', 'secondary_diag', 'additional_diag']

train[i] = train[i].astype('object')

# Frequency of each value in categorical columns
for column in i:
    print(f"Value counts for {column}:")
    display = train[column].value_counts().to_frame()
    print(display, "\n")


# In[62]:


# List the numerical feautres except readmitted col
num_col = [col for col in train.select_dtypes(include=['number']).columns if col != 'readmitted']

# Calculate and print the count of numerical features
num_feature_count = len(num_col)
print(f"Count of numerical features (excluding 'readmitted'): {num_feature_count}")

# Display the list of numerical features
print("Numerical features:", num_col)


# In[63]:


# Plot scatter plot to Visualise the distribution of numerical variables
sns.set()
cols = ['num_med',
 'number_emergency',
 'num_lab_procedures',
 'patient_service',
 'time_in_hospital',
 'med_change',
 'num_procedures',
 'number_diagnoses',
 'number_outpatient',
 'num_medications',
 'number_inpatient']

sns.pairplot(train[cols], height = 2.5)
plt.show();


# In[64]:


# Finding the skewness and kurtosis of the variables 
# List of variables to check
i = ['num_med', 'number_emergency', 'num_lab_procedures', 'patient_service', 'time_in_hospital',
             'med_change', 'num_procedures', 'number_diagnoses', 'number_outpatient', 'num_medications',
             'number_inpatient']

# Print skewness
print("Skewness:")
print(train[i].skew())

# Print kurtosis
print("\nKurtosis:")
print(train[i].kurt())


# In[65]:


# The analysis of skewness and kurtosis reveals that several numerical variables in the dataset
# show significant skewness and/or high kurtosis, indicating distancing from normal distribution. 

# Skewness thresholds:
# - Highly skewed: Skewness less than -1 or greater than 1.
# - Moderately skewed: Skewness between -1 and -0.5 or between 0.5 and 1.
# - Approximately symmetric: Skewness between -0.5 and 0.5.

# Kurtosis thresholds:
# - High kurtosis (> 3) indicates heavy tails and a peaked distribution with more outliers.
# - Kurtosis close to 3 indicates a distribution similar to the normal distribution.
# - Low kurtosis (< 3) suggests light tails and fewer outliers.

# Based on skewness and kurtosis thresholds,the variables requiring transformation are:
# 1. number_emergency (high skewness and very high kurtosis)
# 2. number_outpatient (high skewness and very high kurtosis)
# 3. number_inpatient (high skewness and very high kurtosis)
# 4. patient_service (high skewness and very high kurtosis)
# 5. time_in_hospital (moderately skewed and slightly high kurtosis)
# 6. med_change (moderately skewed and moderately high kurtosis)
# 7. num_procedures (moderately skewed and slightly above normal kurtosis)
# 8. num_medications (moderately skewed and high kurtosis)


# In[66]:


# Log transformation for the skewed numerical variables
key = ['num_med', 'number_emergency', 'num_lab_procedures', 'patient_service', 'time_in_hospital',
               'med_change', 'num_procedures', 'number_diagnoses', 'number_outpatient', 'num_medications',
               'number_inpatient']

for col in key:
    # Apply log transformation to skewed columns and print new skewness values
    if abs(train[col].skew()) >= 1:
        if train[col].min() >= 0:
            train[col + "_log"] = np.log1p(train[col])
            print([col + "_log"], train[col + "_log"].skew())
        else:
            print(f'Column {col} contains negative values cannot be log transformed')

# Standardisation
def standardize(data):
    # Standardizes each column to have mean 0 and standard deviation 1.
    std_dev = np.std(data, axis=0)
    std_dev[std_dev == 0] = 1
    mean = np.mean(data, axis=0)
    return (data - mean) / std_dev

# Applying standardization to identified numeric features
train[num_col] = standardize(train[num_col])

# Correlation Analysis
# Remove columns 'unnamed', case-insensitive from 'train', if there are columns without headers
train = train.drop(columns=train.columns[train.columns.str.contains('unnamed', case=False)])

# Calculate correlation matrix for numeric columns
train_col = train.corr(numeric_only=True)
print(train_col)


# In[67]:


# Find the top 20 correlated variables "readmitted"
k = 20
cols = train_col.nlargest(k,'readmitted')['readmitted'].index 
cm = np.corrcoef(train[cols].values.T)  # Calculates the correlation matrix.
sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(25, 15))
# heatmap with the correlation matrix
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, ax=ax)
plt.show()


# In[68]:


# Transform 'diabetesMed' and 'change' columns type
train.diabetesMed = train.diabetesMed.astype('int64')
train.change = train.change.astype('int64')


# In[69]:


# List of features that need processing.
i = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
     'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
     'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
     'metformin-pioglitazone', 'A1Cresult']

# Convert the columns to float to handle NaN values, then convert to int64
for column in i:
    train[column] = train[column].astype(float).fillna(0).astype('int64')

# Check the data types of the dataframe after transofrmation.
train.dtypes


# In[70]:


# Outliers
key = ['num_med', 'number_emergency', 'num_lab_procedures', 'patient_service', 'time_in_hospital', 'med_change', 
       'num_procedures', 'number_diagnoses', 'number_outpatient', 'num_medications', 'number_inpatient']

plt.figure(figsize=(20, 15))

# Boxplots for each column in 'key'
for i, col in enumerate(key):
    plt.subplot(len(key) // 3 + 1, 3, i + 1)
    sns.boxplot(y=train[col])
    plt.title(col)

plt.tight_layout()
plt.show()


# In[71]:


# Dictionary to hold the outlier values for each column
outliers_dict = {}

for col in key:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter and save the actual outlier values
    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)][col]
    outliers_dict[col] = outliers

# Print out the outliers for each column
for col, values in outliers_dict.items():
    print(f"Outliers in {col}:")
    print(values)


# In[72]:


# Outliers
key = ['num_med', 'number_emergency', 'num_lab_procedures', 'patient_service', 'time_in_hospital', 'med_change', 
       'num_procedures', 'number_diagnoses', 'number_outpatient', 'num_medications','number_inpatient']

# Using IQR to filter out outliers in each column in key
for col in key:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train[col] >= lower_bound) & (train[col] <= upper_bound)]

# Display the names of all columns in the 'train' DataFrame.
train.columns


# In[73]:


# Count of unique values
for i in train.columns:
    df = train[i].value_counts()
    print(df)


# In[74]:


# Convert 'primary_diag' column values to integers for consistency.
train['primary_diag'] = train['primary_diag'].astype('int')
train['secondary_diag'] = train['secondary_diag'].astype('int')
train['additional_diag'] = train['additional_diag'].astype('int')


# In[75]:


# Creating dummy variables for the categorical columns and removing first level do avoid dummy trap
train_v = pd.get_dummies(train, columns=['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'primary_diag', 'secondary_diag', 'additional_diag'], drop_first=True)
print(train_v.columns.tolist())


# In[76]:


# List of categorical columns to be processed.
nom_cols = ['age','race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'primary_diag','secondary_diag', 'additional_diag' ]

# Generate a list of the numerical columns
num_cols = list(set(list(train._get_numeric_data().columns))- {'readmitted', 'change'})
num_cols


# In[77]:


# Initialise list for new column names from dummy variables
nom_cols_new = []
for i in nom_cols:
    for j in train_v.columns:
        if i in j:
            nom_cols_new.append(j)

nom_cols_new


# In[78]:


# Define feature set for the models: Including encoded categories, processed numerical variables, and log-transformed variables.

feature_set = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 
 'diabetesMed', 'patient_service', 'med_change', 'num_med', 'number_emergency_log', 'patient_service_log', 'time_in_hospital_log', 
 'med_change_log', 'num_procedures_log', 'number_outpatient_log', 'num_medications_log', 'number_inpatient_log', 'race_1', 'race_2', 'race_3', 
 'race_4', 'gender_1', 'admission_type_id_3', 'admission_type_id_4', 'admission_type_id_5', 'discharge_disposition_id_2', 
 'discharge_disposition_id_7', 'discharge_disposition_id_10', 'discharge_disposition_id_18', 'discharge_disposition_id_27', 
 'discharge_disposition_id_28', 'admission_source_id_4', 'admission_source_id_7', 'admission_source_id_8', 'admission_source_id_9', 
 'admission_source_id_11', 'A1Cresult_1', 'primary_diag_1', 'primary_diag_2', 'primary_diag_3', 'primary_diag_4', 
 'primary_diag_5', 'primary_diag_6', 'primary_diag_7', 'primary_diag_8', 'secondary_diag_1', 'secondary_diag_2', 'secondary_diag_3', 
 'secondary_diag_4', 'secondary_diag_5', 'secondary_diag_6', 'secondary_diag_7', 'secondary_diag_8', 'additional_diag_1', 'additional_diag_2', 
 'additional_diag_3', 'additional_diag_4', 'additional_diag_5', 'additional_diag_6', 'additional_diag_7', 'additional_diag_8']


# ## Modeling

# In[79]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
sns.set(style='white', context='notebook', palette='deep')

# Splitting the data into features and target variable
train_input = train_v[feature_set]
train_output = train_v['readmitted']

# Check the target variable proportions
target_count=train_v['readmitted'].value_counts()
print('Class 0 (Not Readmitted):', target_count[0])
print('Class 1 (Readmitted):', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Readmitted vs Not Readmitted')

X = train_v.drop('readmitted', axis=1)
y = train_v['readmitted']

# Split the training and testing dataset 
x_train, x_test, y_train, y_test= model_selection.train_test_split(train_input, train_output, random_state = 0, test_size=0.3)


# ## Logistic Regression (Baseline)

# In[80]:


for column in train_v.select_dtypes(include=['object']).columns:
    # Convert object columns to categorical and replace with their integer codes for logistic regression model compatibility
    train_v[column] = train_v[column].astype('category').cat.codes

# Split the data (70% training, 30% testing)
X = train_v.drop('readmitted', axis=1)  # Features
y = train_v['readmitted']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=10000)

# Fit the model
logreg.fit(X_train, y_train)

# Predict the outcomes on the testing data
y_pred = logreg.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Cross Validation
cv_scores = cross_val_score(logreg, X, y, cv=10)
mean_cv_score = cv_scores.mean()

# Print performance metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {mean_cv_score:.4f}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualise the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Readmitted', 'Readmitted'],
            yticklabels=['Not Readmitted', 'Readmitted'])
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# ## Balancing

# In[81]:


# Separate majority and minority classes
df_majority = train_v[train_v.readmitted==0]
df_minority = train_v[train_v.readmitted==1]

print("Initial sizes and proportions:")
print("Class 0 (Not Readmitted):", len(df_majority))
print("Class 1 (Readmitted):", len(df_minority))
print("Proportion:", round(len(df_majority) / len(df_minority), 2), ": 1")

# Oversample minority class
df_minority_oversampled = df_minority.sample(n=len(df_majority), replace=True, random_state=0)
df_oversampled = pd.concat([df_majority, df_minority_oversampled])

# Splitting the oversampled dataset into input and output
train_input_new = df_oversampled[feature_set]
train_output_new = df_oversampled['readmitted']

# Print sizes and proportions after balancing
print("\nSizes and proportions after balancing:")
print("Class 0 (Not Readmitted):", len(df_majority))
print("Class 1 (Readmitted) after oversampling:", len(df_minority_oversampled))
print("Proportion after balancing:", "1 : 1")

# Split the data (70% training, 30% testing)
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(train_input_new, train_output_new, test_size=0.3, random_state=0)


# ## RandomForest

# In[82]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Split the balanced data (70% training, 30% testing)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(train_input_new, train_output_new, test_size=0.3, random_state=0)

# Initialize RandomForest classifier
random_forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)

# Fit the model on the balanced training data
random_forest.fit(X_train_new, y_train_new)

# Predict the outcomes on the balanced testing data
y_pred_rf = random_forest.predict(X_test_new)

# Performance metrics for the balanced dataset
accuracy_rf = accuracy_score(y_test_new, y_pred_rf)
precision_rf = precision_score(y_test_new, y_pred_rf)
recall_rf = recall_score(y_test_new, y_pred_rf)

# Print performance metrics
print(f'RandomForest - Balanced Data - Accuracy: {accuracy_rf:.4f}')
print(f'RandomForest - Balanced Data - Precision: {precision_rf:.4f}')
print(f'RandomForest - Balanced Data - Recall: {recall_rf:.4f}')

# Generate the confusion matrix
conf_matrix_rf = confusion_matrix(y_test_new, y_pred_rf)

# Visualise the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Readmitted', 'Readmitted'],
            yticklabels=['Not Readmitted', 'Readmitted'])
plt.title('Confusion Matrix for RandomForest on Balanced Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Perform cross-validation and calculate scores
cv_scores_rf = cross_val_score(random_forest, X_train_new, y_train_new, cv=10)

# Calculate the mean of the cross-validation scores
cv_scores_mean_rf = cv_scores_rf.mean()

# Print the cross-validation scores for each fold
print("Cross-validation scores:", cv_scores_rf)

# Print the mean of the cross-validation scores
print(f"Mean cross-validation score for RandomForest: {cv_scores_mean_rf:.4f}")

# Print the OOB score
print(f'RandomForest - OOB Score: {random_forest.oob_score_:.4f}')

# Classification report
classification_report_rf = classification_report(y_test_new, y_pred_rf)
print("Classification Report for RandomForest on Balanced Data:\n", classification_report_rf)


# ## Feature Importance

# In[83]:


# Generate and display feature importances
feature_importances = random_forest.feature_importances_
features = X_train_new.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
pd.set_option('display.max_rows', None)
print(importance_df)
pd.reset_option('display.max_rows')


# ## Feature Set 2

# In[84]:


# Selecting 23 most important features filtered
feature_set2 = [
    "num_lab_procedures",
    "diag_1",
    "diag_2",
    "diag_3",
    "num_medications_log",
    "num_medications",
    "age",
    "number_diagnoses",
    "time_in_hospital",
    "time_in_hospital_log",
    "num_procedures",
    "num_procedures_log",
    "gender_1",
    "num_med",
    "admission_source_id_7",
    "discharge_disposition_id_2",
    "additional_diag_1",
    "race_1",
    "secondary_diag_1",
    "insulin",
    "race_2",
    "primary_diag_1",
    "A1Cresult_1"
]


# In[85]:


train_input_new = df_oversampled[feature_set2]
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(train_input_new, train_output_new, test_size=0.3, random_state=0)


# ## RandomForest 2

# In[86]:


# Split the balanced data into training and testing sets (70% training, 30% testing)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(train_input_new, train_output_new, test_size=0.3, random_state=0)

# Initialize RandomForest classifier
random_forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)

# Fit the model on the balanced training data
random_forest.fit(X_train_new, y_train_new)

# Predict the outcomes
y_pred_rf = random_forest.predict(X_test_new)

# Performance metrics for the balanced dataset
accuracy_rf = accuracy_score(y_test_new, y_pred_rf)
precision_rf = precision_score(y_test_new, y_pred_rf)
recall_rf = recall_score(y_test_new, y_pred_rf)

# Print performance metrics for the balanced dataset
print(f'RandomForest - Balanced Data - Accuracy: {accuracy_rf:.4f}')
print(f'RandomForest - Balanced Data - Precision: {precision_rf:.4f}')
print(f'RandomForest - Balanced Data - Recall: {recall_rf:.4f}')

# Generate the confusion matrix for the balanced dataset
conf_matrix_rf = confusion_matrix(y_test_new, y_pred_rf)

# Visualise the confusion matrix for the balanced dataset
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Readmitted', 'Readmitted'],
            yticklabels=['Not Readmitted', 'Readmitted'])
plt.title('Confusion Matrix for RandomForest on Balanced Data')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Perform cross-validation and calculate scores
cv_scores_rf = cross_val_score(random_forest, X_train_new, y_train_new, cv=10)

# Calculate the mean of the cross-validation scores
cv_scores_mean_rf = cv_scores_rf.mean()

# Print the cross-validation scores for each fold
print("Cross-validation scores for each fold:", cv_scores_rf)

# Print the mean of the cross-validation scores
print(f"Mean cross-validation score for RandomForest: {cv_scores_mean_rf:.4f}")

# Print the OOB score
print(f'RandomForest - OOB Score: {random_forest.oob_score_:.4f}')

# Classification report
classification_report_rf = classification_report(y_test_new, y_pred_rf)
print("Classification Report:\n", classification_report_rf)


# ## K-means Clustering

# In[87]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Selecting the features for KMeans
feature_Kmeans = [
    "num_lab_procedures",
    "num_medications_log",
    "num_medications",
    "age",
    "time_in_hospital",
]

train_input_new = train_input_new[feature_Kmeans]

# Normalisation of the features
train_input_new = (train_input_new - train_input_new.min()) / (train_input_new.max() - train_input_new.min())


# In[88]:


# Determine the optimal number of clusters using the Elbow Method
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=0).fit(train_input_new)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[89]:


# Optimal number of clusters is 4 based on the elbow method
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, max_iter=1000, random_state=0).fit(train_input_new)
clusters = kmeans.labels_
train_input_new['cluster'] = clusters


# In[90]:


# PCA visualization
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_input_new.drop('cluster', axis=1))
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Combine PCA and cluster labels
finalDf = pd.concat([principalDf, train_input_new[['cluster']].reset_index(drop=True)], axis=1)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA')

colors = ['r', 'g', 'b', 'y', 'c', 'm']
for cluster in range(optimal_k):
    cluster_data = finalDf[finalDf['cluster'] == cluster]
    plt.scatter(cluster_data['principal component 1'], 
                cluster_data['principal component 2'], 
                c=colors[cluster], 
                s=5)
plt.show()

# Grouping the data by the 'cluster' column and calculating the mean for each group
cluster_means = train_input_new.groupby('cluster').mean()
print(cluster_means)


# In[91]:


# Bar plot visualizing the means for each feature by cluster.
cluster_means.plot(kind='bar', figsize=(15, 7))
plt.title('Feature Means by Cluster')
plt.ylabel('Mean Value')
plt.xlabel('Cluster')
plt.xticks(rotation=0) 
plt.legend(title='Features')
plt.show()

