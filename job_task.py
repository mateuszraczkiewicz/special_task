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
histogram(przychod, axes, 0, 1, 30, 'cyan', 'Histogram przychodu klientów', 'PRzychód')
histogram(czas_na_stronie, axes, 1, 0, 30,'lime', 'Histogram czasu na stronie klienta', 'Czas na stronie [min]')
histogram(liczba_wizyt, axes, 1, 1, 20, 'springgreen', 'Histogram lyczby wizyt klientów', 'Liczba klientów')


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

plt.title('Udział klientów: kupił vs. nie kupił', fontsize=14, fontweight='bold')



# Podział na kolumny X i y
X = df.loc[:, df.columns != 'kupil']
y = kupil
print(X)
print(y)
print(kupil.value_counts())

# Widzmy że dane są niezbalansowane. Większa część decyzji to nie kupienie usług. By poprwaić detekcję modelu wykorzystamy oversampling, by powiększyć zbiór 'kupil'
ros = RandomUnderSampler(sampling_strategy=1)
# rus = RandomUnderSampler(sampling_strategy=1) - dla under samplingu wyniki dla RFC i XGB są gorsze 
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

plt.title('Udział klientów: kupił vs. nie kupił (po undersamplingu)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Trening modelu
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predykcja
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# === Klasyfikacja: metryki ===
print("=== TRAIN SET ===")
print(classification_report(y_train, y_train_pred))
print("ROC AUC (train):", roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]))

print("\n=== TEST SET ===")
print(classification_report(y_test, y_test_pred))
print("ROC AUC (test):", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# === ROC Curve ===
fpr_train, tpr_train, _ = roc_curve(y_train, rf.predict_proba(X_train)[:, 1])
fpr_test, tpr_test, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 5))
plt.plot(fpr_train, tpr_train, label='Train ROC Curve', linestyle='--', color='blue')
plt.plot(fpr_test, tpr_test, label='Test ROC Curve', color='green')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
# # Model 1 - Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred_logreg = logreg.predict(X_test)

# # print("===== Logistic Regression =====")
# # print(classification_report(y_test, y_pred_logreg))
# # print("ROC AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))

# # # Model 2 - Random Forest
# # rf = RandomForestClassifier(n_estimators=100, random_state=42)
# # rf.fit(X_train, y_train)
# # y_pred_rf = rf.predict(X_test)

# # print("\n===== Random Forest Classifier =====")
# # print(classification_report(y_test, y_pred_rf))
# # print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# # # Model 3 - XGBoost
# # xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
# # xgb.fit(X_train, y_train)
# # y_pred_xgb = xgb.predict(X_test)

# # print("\n===== XGBoost Classifier =====")
# # print(classification_report(y_test, y_pred_xgb))
# # print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))
