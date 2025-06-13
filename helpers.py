import numpy as np
from itertools import combinations
import pandas as pd


# funkcja stosująca regułę
def apply_rule(data, rule):
    # tworzenie kopii danych wejściowych, od których będziemy odfiltrowywać przypadki.
    filtered_data = data

    # Iterujemy przez warunki reguły, każdy w postaci (atrybut, wartość).
    for feature, value, in rule:
        # filtrowanie wierszy, które spełniają warunek feature == value.
        filtered_data = filtered_data[filtered_data[feature] == value]

    # pobieranie unikalnych wartości klasy decyzyjnej
    unique_values = filtered_data["Decyzja"].unique()

    # sprawdzanie, czy dana reguła nie jest sprzeczną.
    if len(unique_values) == 1:
        return filtered_data
    else:
        return pd.DataFrame(
            columns=data.columns
        )  # jeśli mamy sprzeczność, to zwracamy pusty data frame.


def sequential_covering(X, y, max_rules=5):
    # tworzenie kopii danych zawierającej cechy oraz klasę decyzyjna
    data = X.copy()
    data["Decyzja"] = y

    rules = []  # lista końcowa reguł decyzyjnych
    unique_rules = set()  # set przechowuje tylko wartości unikalne, unikamy duplikatów reguł decyzyjnych.
    remaining_data = data.copy()

    # krok pierwszy: generowanie reguł 1-rzędu
    for index, row in remaining_data.iterrows():
        # przechodzimy po każdej cesze (oprócz kolumny decyzyjnej).
        for feature in remaining_data.columns[:-1]:
            value = row[feature]
            rule = frozenset([(feature, value)])

            # jeśli reguła nie była wcześniej rozważana.
            if rule not in unique_rules:
                # chcemy zastosować regułę na pełnym zbiorze danych.
                covered = apply_rule(data, list(rule))

                # jeśli reguła pokrywa jakiekolwiek przykłady.
                if not covered.empty:
                    # sprawdzamy jakie klasy występują w pokrytych przykładach.
                    unique_classes = covered["Decyzja"].unique()

                    if len(unique_classes) == 1:
                        rule_class = unique_classes[0]

                        # dodajemy regułę do list końcowej.
                        rules.append((list(rule), rule_class, len(covered)))
                        unique_rules.add(rule)  # oznaczamy regułę jako używaną.

                        # usuwanie przykładów pokrytych
                        remaining_data = remaining_data.drop(covered.index, errors="ignore")

                        if len(rules) >= max_rules:
                            return rules

    # przejście do reguł decyzyjnych 2-rzędu
    for index, row in remaining_data.iterrows():

        marker = False  # flaga pomocnicza czy utworzono regułę dla danego wiersza.

        for feature_pair in combinations(remaining_data.columns[:-1], 2):

            feature1, feature2 = feature_pair
            value1 = row[feature1]
            value2 = row[feature2]

            rule = frozenset([(feature1, value1), (feature2, value2)])

            if rule not in unique_rules:
                covered = apply_rule(data, list(rule))

                # sprawdzamy sprzeczność danej reguły.
                if not covered.empty:
                    unique_classes = covered["Decyzja"].unique()

                    # Reguła jest akceptowana, tylko jeśli przypisuje jednoznacznie jedną klasę.
                    if len(unique_classes) == 1:
                        rule_class = unique_classes[0]

                        rules.append((list(rule), rule_class, len(covered)))
                        unique_rules.add(rule)
                        remaining_data = remaining_data.drop(covered.index, errors="ignore")
                        marker = True

                        if len(rules) >= max_rules:
                            return rules

                        break  # przerywamy pętlę wewnętrzną.

    # zwracamy zestaw wygenerowanych reguł.
    return rules


# funkcja predykcji nowych danych
def predict(X, rules):
    predictions = []
    for _, row in X.iterrows():
        prediction = 0 # domyślna klasa negatywna
        for rule, rule_class, _ in rules:
            if all(row[feature] == value for feature, value in rule):
                prediction = rule_class
                break
        predictions.append(prediction)
    return np.array(predictions)
