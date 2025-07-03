import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
# wiek, przychod, czas_na_stronie_min, liczba_wizyt, godzina_wejscia, dzien_tygodnia – zmienne numeryczne
# typ_klienta, region – zmienne kategoryczne
# natomiast kupli - będzie zmienną docelową to ją chcemy przewidywać 

# Eksploracja danych

# zmienne numeryczne
data = {
    'średnia': [df.iloc[:, i].mean() for i in range(6)],
    'max': [df.iloc[:, i].max() for i in range(6)],
    'min': [df.iloc[:, i].min() for i in range(6)]
}

rows = ['wiek', 'przychod', 'czas_na_stronie_min', 'liczba_wizyt', 'godzina_wejscia', 'dzien_tygodnia']
df_numeric_var = pd.DataFrame(data, index=rows)
print(df_numeric_var)

wiek = df['wiek']
przychod = df['przychod']
czas_na_stronie = df['czas_na_stronie_min']
liczba_wizyt = df['liczba_wizyt']
godzina_wejscia = df['godzina_wejscia']
dzien_tygodnia = df['dzien_tygodnia']
zakup = df['kupil']

plt.style.use('seaborn-v0_8')
# Histogramy
fig1, axes = plt.subplots(3, 2, figsize=(14, 9))

histogram(wiek, axes, 0, 0, 30, 'skyblue', 'Histogram wieku klientów', 'Wiek klientów')
histogram(przychod, axes, 0, 1, 30, 'cyan', 'Histogram przychodu klientów', 'PRzychód')
histogram(czas_na_stronie, axes, 1, 0, 30,'lime', 'Histogram czasu na stronie klienta', 'Czas na stronie [min]')
histogram(liczba_wizyt, axes, 1, 1, 20, 'springgreen', 'Histogram lyczby wizyt klientów', 'Liczba klientów')
histogram(godzina_wejscia, axes, 2, 0, 24, 'salmon', 'Histogram godzina wejścia kilentów', 'Godziny wejścia klientów')
histogram(dzien_tygodnia, axes, 2, 1, 7, 'coral', 'Histogram dnia tygodnia odwiedzania storny przez klienta', 'Dzień tygodnia')

plt.tight_layout()
plt.show()

# Wykresy słupkowe
fig2, axes = plt.subplots(3, 2, figsize=(14, 9))

barplot(df, 'wiek', axes, 0, 0, 'skyblue')
barplot(df, 'przychod', axes, 0, 1, 'cyan', mean=100, width=100)
barplot(df, 'czas_na_stronie_min', axes, 1, 0, 'lime')
barplot(df, 'godzina_wejscia', axes, 1, 1, 'springgreen')
barplot(df, 'dzien_tygodnia', axes, 2, 0, 'salmon')
barplot(df, 'liczba_wizyt', axes, 2, 1, 'coral')


plt.tight_layout()
plt.show()


# Zmienne kategoryczne
# W tej tabeli mamy 2 kolumny zawierające dane kateogryczne są to typy kilenta i region
# Obie zawierją stringi z informacjami, informacje te nie są do ustawienia w logicznej kolejności 
dane_kategoryczne = df[['typ_klienta', 'region']]
print(dane_kategoryczne.head(5))

# Sprawdzamy unikalne wartości
print(dane_kategoryczne['typ_klienta'].unique())
print(dane_kategoryczne['region'].unique())

fig1, axes = plt.subplots(2, 2, figsize=(16, 6))

histogram(dane_kategoryczne['typ_klienta'], axes, 0, 0, 30, 'skyblue', 'Histogram typów klientów', 'Typ klientów')
histogram(dane_kategoryczne['region'], axes, 0, 1, 30, 'cyan', 'Histogram regionu pochodzenia klientów', 'Region')
barplot(df, 'typ_klienta', axes, 1, 0, 'lime')
barplot(df, 'region', axes, 1, 1, 'springgreen')

plt.tight_layout()
plt.show()

# Enkodowanie danych kategorycznych 
df = pd.get_dummies(df, columns=['typ_klienta', 'region'], dtype=int)

# Enkodowanie danych numerycznych z użyciem min max normalizacji
normalizer = MinMaxScaler()
wyniki = normalizer.fit_transform(df[['wiek', 'przychod', 'czas_na_stronie_min', 'liczba_wizyt']].values.reshape(-1, 1))
print(wyniki)

# Podział na kolumny X i y
X = df.loc[:, df.columns != 'kupil']
y = df['kupil']

# 5. Podział danych
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
