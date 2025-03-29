import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Am folosit pandas pentru a incarca dataset-ul despre Titanic direct din seaborn
titanic = sns.load_dataset('titanic')

# Afisam primele cateva linii si coloane
print(titanic.head())

# Informatii basic despre dataset
print("\nDataset Info:")
titanic.info()

# Rezumat statistic
print("\nStatistical Summary:")
print(titanic.describe())

# Cautam valorile care lipsesc
print("\nMissing Values:")
print(titanic.isnull().sum())

# Creez o copie pentru a evita modificarea originalului
titanic_clean = titanic.copy()

# Ma ocup de missing values din 'age'
# Inlocuiesc missing values cu median age
titanic_clean.fillna({'age':titanic_clean['age'].median()}, inplace=True)

# Ma ocup de missing values din 'embarked'
# Inlocuiesc cu most common embarkation point
most_common_embarked = titanic_clean['embarked'].mode()[0]
titanic_clean.fillna({'embarked':most_common_embarked}, inplace=True)

# Renuntam la coloana 'deck' pentru ca are prea multe valori lipsa
titanic_clean.drop('deck', axis=1, inplace=True)

# In coloana 'embarked', convertesc la numeric pentru numpy operations
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
titanic_clean['embarked_code'] = titanic_clean['embarked'].map(embarked_mapping)

# Convertesc la numpy arrays
survival = titanic_clean['survived'].values
ages = titanic_clean['age'].values
fares = titanic_clean['fare'].values
pclass = titanic_clean['pclass'].values
sex = np.where(titanic_clean['sex'] == 'male', 1, 0)  # Convert to binary

# Calculez rata de supravietuire
survival_rate = np.mean(survival)
print(f"\nOverall survival rate: {survival_rate:.2%}")

# Calculez rata de supravietuire in functie de gen
male_survival = np.mean(survival[sex == 1])
female_survival = np.mean(survival[sex == 0])
print(f"Male survival rate: {male_survival:.2%}")
print(f"Female survival rate: {female_survival:.2%}")

# Calculez rata de supravietuire in functie de clasa
for i in range(1, 4):
    class_survival = np.mean(survival[pclass == i])
    print(f"Class {i} survival rate: {class_survival:.2%}")

# Statistici despre varsta
print(f"\nAge statistics:")
print(f"Mean age: {np.mean(ages):.2f}")
print(f"Median age: {np.median(ages):.2f}")
print(f"Standard deviation: {np.std(ages):.2f}")
print(f"Min age: {np.min(ages):.2f}")
print(f"Max age: {np.max(ages):.2f}")

# Distributia varstei
survived_ages = ages[survival == 1]
died_ages = ages[survival == 0]
print(f"Mean age of survivors: {np.mean(survived_ages):.2f}")
print(f"Mean age of non-survivors: {np.mean(died_ages):.2f}")

# Creez un figure cu subplots
plt.figure(figsize=(15, 10))

# Survival distribution
plt.subplot(2, 2, 1)
sns.countplot(x='survived', data=titanic_clean)
plt.title('Survival Distribution')

# Survival by gender
plt.subplot(2, 2, 2)
sns.countplot(x='sex', hue='survived', data=titanic_clean)
plt.title('Survival by Gender')

# Survival by passenger class
plt.subplot(2, 2, 3)
sns.countplot(x='pclass', hue='survived', data=titanic_clean)
plt.title('Survival by Passenger Class')

# Age distribution
plt.subplot(2, 2, 4)
sns.histplot(titanic_clean['age'], bins=30, kde=True)
plt.title('Age Distribution')

plt.tight_layout()
plt.show()

# Correlation matrix for numeric features
numeric_features = titanic_clean.select_dtypes(include=[np.number])
correlation = np.corrcoef(numeric_features.T)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True,
            xticklabels=numeric_features.columns,
            yticklabels=numeric_features.columns)
plt.title('Correlation Matrix')
plt.show()

# Calculez probabilitatea de supravietuire in functie de grupa de varsta folosind NumPy
age_bins = np.linspace(0, 80, 9)  # Create 8 age groups from 0 to 80
age_groups = np.digitize(ages, age_bins)

for i in range(1, len(age_bins)):
    group_survival = np.mean(survival[age_groups == i])
    print(f"Age {age_bins[i-1]:.0f}-{age_bins[i]:.0f}: Survival rate {group_survival:.2%}")

