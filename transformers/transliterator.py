from transliterate import translit
import cyrtranslit
import pytils

strs = {"Tiberius": "т_trans и_trans б_trans е_trans р_trans и_trans у_trans с_trans",
        "Julius": "д_trans ж_trans у_trans л_trans и_trans у_trans с_trans",
        "Pollienus": "п_trans о_trans л_trans л_trans и_trans е_trans н_trans у_trans с_trans",
        "Auspex": "о_trans с_trans п_trans е_trans к_trans с_trans",
        "Half": "х_trans а_trans л_trans ф_trans",
        "Armor": "а_trans р_trans м_trans о_trans р_trans",
        "Sbrinz": "с_trans б_trans р_trans и_trans н_trans с_trans",
        "Kase": "к_trans е_trans й_trans с_trans",
        "GmbH": "г_trans м_trans б_trans",
        "The": "з_trans э_trans",
        "next": "н_trans е_trans к_trans с_trans т_trans",
        "supermoon": "с_trans у_trans п_trans е_trans р_trans м_trans у_trans н_trans",
        "in": "и_trans н_trans",
        "is": "и_trans с_trans",
        "July": "д_trans ж_trans у_trans л_trans и_trans"
        }

for s in strs:
    w = s.lower()
    print(strs[s].replace('_trans', '').replace(' ', ''),
          translit(w, language_code='ru'),
          cyrtranslit.to_cyrillic(w, lang_code='ru'),
          pytils.translit.detranslify(w))
