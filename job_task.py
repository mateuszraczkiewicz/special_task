import pandas as pd
import matplotlib.pyplot as plt

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
print(wiek.unique())
przychod = df['przychod']
zakup = df['kupil']

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(14, 5))

# Histogram wieku
axes[0, 0].hist(wiek, bins=30, rwidth=0.9, ec='black', color='skyblue')
axes[0, 0].set_title('Histogram wieku klientów')
axes[0, 0].set_xlabel('Wiek')
axes[0, 0].set_ylabel('Liczba klientów')

# Histogram przychodu
axes[0, 1].hist(przychod, bins=30, rwidth=0.9, ec='black', color='salmon')
axes[0, 1].set_title('Histogram przychodu klientów')
axes[0, 1].set_xlabel('Przychód')
axes[0, 1].set_ylabel('Liczba klientów')

# Zlicz ile osób w danym wieku kupiło (kupil == 1)
zakupy_wg_wieku = df[df['kupil'] == 1].groupby('wiek').size()

# Zamiana na DataFrame (opcjonalnie, dla lepszej prezentacji)
zakupy_wg_wieku = zakupy_wg_wieku.reset_index(name='liczba_zakupow')

# Wyświetl
print(zakupy_wg_wieku)
            
axes[1, 1].bar(zakupy_wg_wieku['wiek'], zakupy_wg_wieku['liczba_zakupow'], ec='black', color='green')
axes[1, 1].set_title('Wykres ilości zakupów w zależnośći od wieku')
axes[1, 1].set_xlabel('Wiek')
axes[1, 1].set_ylabel('Liczba zakupów')

plt.tight_layout()
plt.show()