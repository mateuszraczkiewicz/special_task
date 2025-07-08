import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def histogram(data, ax, row_index, column_index, bins, color_of_chart, title, x_label, y_label = 'Liczba klientów'):
    ax[row_index, column_index].hist(data, bins=bins, rwidth=0.9, ec='black', color=color_of_chart)
    ax[row_index, column_index].set_title(title)
    ax[row_index, column_index].set_xlabel(x_label)
    ax[row_index, column_index].set_ylabel(y_label)
    
    
def barplot(data_frame, kategoria, ax, row_index, column_index, color, mean=1, width=1):
    # Zlicz ile osób w danym wieku kupiło (kupil == 1)
    zakupy_wg_kategori = data_frame[data_frame['kupil'] == 1].groupby(kategoria).size()

    # Zamiana na DataFrame (opcjonalnie, dla lepszej prezentacji)
    zakupy_wg_kategori = zakupy_wg_kategori.reset_index(name='liczba_zakupow')
    
    # Uśrednianie
    if mean != 1:
        zakupy_wg_kategori = zakupy_wg_kategori.groupby(np.arange(len(zakupy_wg_kategori))//mean).mean()
                
    ax[row_index, column_index].bar(zakupy_wg_kategori[kategoria], zakupy_wg_kategori['liczba_zakupow'], ec='black', color=color, width = width)
    ax[row_index, column_index].set_title(f'Wykres ilości zakupów w zależnośći od {kategoria}')
    ax[row_index, column_index].set_xlabel(kategoria)
    ax[row_index, column_index].set_ylabel('Liczba zakupów')
    
def Sprawdzanie_przetrenowania(model, X, y):
    # Predykcja
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Klasyfikacja: metryki
    print("Zbiór treningowy")
    print(classification_report(y_train, y_train_pred))
    print("ROC AUC (train):", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))

    print("\nZbiór testowy")
    print(classification_report(y_test, y_test_pred))
    print("ROC AUC (test):", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1])
    fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(8, 5))
    plt.plot(fpr_train, tpr_train, label='Train ROC Curve', linestyle='--', color='blue')
    plt.plot(fpr_test, tpr_test, label='Test ROC Curve', color='green')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Zczytanie danych i pierwsze spojrzenie na dane
df = pd.read_csv('dane_zakupowe.csv', sep=';')
print(df.head(5))
print(df.info())

 
# Na pierwszy rzut oka widzimy że dane : 
# wiek, przychod, czas_na_stronie_min, liczba_wizyt – zmienne numeryczne
# godzina_wejscia, dzien_tygodnia - są podane w liczbach ale będziemy je traktowac jako zmienne kategoryczne 
# typ_klienta, region – zmienne kategoryczne
# natomiast kupli - będzie zmienną docelową to ją chcemy przewidywać 

# Eksploracja danych

# zmienne numeryczne
data = {
    'średnia': [df.iloc[:, i].mean() for i in range(4)],
    'max': [df.iloc[:, i].max() for i in range(4)],
    'min': [df.iloc[:, i].min() for i in range(4)]
}

rows = ['wiek', 'przychod', 'czas_na_stronie_min', 'liczba_wizyt']
df_numeric_var = pd.DataFrame(data, index=rows)
print(df_numeric_var)

wiek = df['wiek']
przychod = df['przychod']
czas_na_stronie = df['czas_na_stronie_min']
liczba_wizyt = df['liczba_wizyt']

plt.style.use('seaborn-v0_8')
# Histogramy
Histogramy, axes = plt.subplots(2, 2, figsize=(14, 9))

histogram(wiek, axes, 0, 0, 30, 'skyblue', 'Histogram wieku klientów', 'Wiek klientów')
histogram(przychod, axes, 0, 1, 30, 'cyan', 'Histogram przychodu klientów', 'Przychód klientów')
histogram(czas_na_stronie, axes, 1, 0, 30,'lime', 'Histogram czasu na stronie klienta', 'Czas na stronie klientów [min]')
histogram(liczba_wizyt, axes, 1, 1, 20, 'springgreen', 'Histogram liczby wizyt klientów', 'Liczba wizyt klientów')


plt.tight_layout()
plt.show()

# Wykresy słupkowe
Wykresy_slupkowe, axes = plt.subplots(2, 2, figsize=(14, 9))

barplot(df, 'wiek', axes, 0, 0, 'skyblue')
barplot(df, 'przychod', axes, 0, 1, 'cyan', mean=100, width=100)
barplot(df, 'czas_na_stronie_min', axes, 1, 0, 'lime')
barplot(df, 'liczba_wizyt', axes, 1, 1, 'springgreen')



plt.tight_layout()
plt.show()


# Zmienne kategoryczne
# W tej tabeli mamy 4 kolumny zawierające dane kateogryczne są to typy kilenta, godzina_wejscia, dzien_tygodnia i region
# Typ klienta i region zawierją stringi z informacjami, informacje te nie są do ustawienia w logicznej kolejności
# Godzina_wejscia i Dzien_tygodnia są podawane w wartościach liczbowych ale dalej każda z tych liczb to oddzielna kategoria.
# Potrakotowanie ich jako liczbe mogłoby stwarzać problemy gdyż program by mógł uznać że 23 jest ważniejsza od 10 bo ma większą wartość. Ze względu na to potrakutjemy je hot line encoding 
dane_kategoryczne = df[['typ_klienta', 'region', 'godzina_wejscia', 'dzien_tygodnia']]
print(dane_kategoryczne.head(5))
typ_klienta = dane_kategoryczne['typ_klienta']
region = dane_kategoryczne['region']
godzina_wejscia = dane_kategoryczne['godzina_wejscia']
dzien_tygodnia = dane_kategoryczne['dzien_tygodnia']
# Sprawdzamy unikalne wartości
print(f'Unikalne wartości dla typu klienta - {typ_klienta.unique()}\n')
print(f'Unikalne wartości dla regiou - {region.unique()}\n')
print(f'Unikalne wartości dla godziny wejścia - {np.sort(godzina_wejscia.unique())}\n')
print(f'Unikalne wartości dla dnia tygodnia - {np.sort(dzien_tygodnia.unique())}\n')

Histogramy_i_wykresy_slupkowe, axes = plt.subplots(2, 4, figsize=(20, 8))

histogram(typ_klienta, axes, 0, 0, 30, 'skyblue', 'Histogram typów klientów', 'Typ klientów')
histogram(region, axes, 0, 1, 30, 'cyan', 'Histogram regionu pochodzenia klientów', 'Region')
histogram(godzina_wejscia, axes, 0, 2, 24, 'salmon', 'Histogram godzina wejścia kilentów', 'Godziny wejścia klientów')
histogram(dzien_tygodnia, axes, 0, 3, 7, 'coral', 'Histogram dnia tygodnia odwiedzania storny przez klienta', 'Dzień tygodnia')
barplot(df, 'typ_klienta', axes, 1, 0, 'lime')
barplot(df, 'region', axes, 1, 1, 'springgreen')
barplot(df, 'godzina_wejscia', axes, 1, 2, 'salmon')
barplot(df, 'dzien_tygodnia', axes, 1, 3, 'coral')

plt.tight_layout()
plt.show()

# Enkodowanie danych kategorycznych 
df = pd.get_dummies(df, columns=['typ_klienta', 'region', 'godzina_wejscia', 'dzien_tygodnia'], dtype=int)

# Enkodowanie danych numerycznych z użyciem min max normalizacji
normalizer = MinMaxScaler()
df[['wiek', 'przychod', 'czas_na_stronie_min', 'liczba_wizyt']] = normalizer.fit_transform(df[['wiek', 'przychod', 'czas_na_stronie_min', 'liczba_wizyt']])
print(df.head(5))

# Zmienna docelowa
kupil = df['kupil']
Wykresy_tortowe, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].pie(kupil.value_counts(),
        labels=['Nie kupił', 'Kupił'],
        colors=['coral', 'lime'],
        autopct='%.1f%%',
        explode=(0.03, 0.08),
        startangle=140,
        shadow=True,
        wedgeprops={'edgecolor': 'white'})
axes[0].set_title('Udział klientów: kupił vs. nie kupił')

# Podział na kolumny X i y
X = df.loc[:, df.columns != 'kupil']
y = kupil
print(X)
print(y)
print(kupil.value_counts())

# Widzmy że dane są niezbalansowane. Większa część decyzji to nie kupienie usług. By poprwaić detekcję modelu wykorzystamy oversampling, by powiększyć zbiór 'kupil'
ros = RandomOverSampler(sampling_strategy=1)
# ros = RandomUnderSampler(sampling_strategy=1) - dla under samplingu wyniki dla RFC i XGB są gorsze 
X_res, y_res = ros.fit_resample(X, y)
print(X_res)
print(y_res)
print(y_res.value_counts())


axes[1].pie(y_res.value_counts(),
        labels=['Nie kupił', 'Kupił'],
        colors=['coral', 'lime'],
        autopct='%.1f%%',
        explode=(0.03, 0.08),
        startangle=140,
        shadow=True,
        wedgeprops={'edgecolor': 'white'})
axes[1].set_title('Udział klientów: kupił vs. nie kupił (po oveersamplingu)')


plt.tight_layout()
plt.show()

# 5. Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


# Model 1 - Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Regresja Logiczna")
print(classification_report(y_test, y_pred_logreg))
print("ROC AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))

# Model 2 - Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, min_samples_leaf=1, min_samples_split=5)
# rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forsest")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))


# Model 3 - XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost")
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

# Sprawdzenie przetrenowania do RandomForest
# Sprawdzanie_przetrenowania(rf, X_res, y_res)

# # Parametry do tuningu
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 5],
#     'max_features': ['sqrt', 'log2']
# }

# # Random Forest
# rf = RandomForestClassifier(random_state=42)

# # Grid Search z walidacją krzyżową
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=param_grid,
#                            scoring='roc_auc',
#                            cv=3,
#                            n_jobs=-1,
#                            verbose=2)

# # Trening
# grid_search.fit(X_train, y_train)

# # Najlepszy model
# best_rf = grid_search.best_estimator_
# print("Najlepsze parametry:", grid_search.best_params_)

# # Ocena na zbiorze testowym
# y_pred = best_rf.predict(X_test)
# y_proba = best_rf.predict_proba(X_test)[:, 1]

# print("\nTEST SET")
# print(classification_report(y_test, y_pred))
# print("ROC AUC:", roc_auc_score(y_test, y_proba))




