import xml.etree.ElementTree as ET
import re
import os
import numpy as np

#file paths
data_path = r'C:\Users\jonas\OneDrive\Dokumente\Master Data Science\1. Semester\Digital Humanities\project\data'
input_folder = data_path + r"\Wahlperiode 20"  # Ordner mit Protokolldateien
output_folder = data_path + r"\reden_die_linke"  # Zielordner für die extrahierten Dateien

def extract_speeches(root):
    """
    extracts the speeches from speakers of the fraction 'Die Linke'
    :param root: root of .xml tree
    :return: list of strings consisting of 'Die Linke' speeches
    """
    # list of speeches for "Die Linke"
    linke_speeches = []

    for speech in root.findall(".//rede"):
        speaker = speech.find(".//redner")
        fraction = speaker.find("name/fraktion")
        if fraction is not None and (fraction.text == "Die Linke" or fraction.text == "DIE LINKE"):
            speech_text = ''.join(speech.itertext()).strip()
            #removes the speaker name and fraction from the text
            match = re.search(r':', speech_text)
            if match:
                speech_text = speech_text[match.start() + 1:]
                if speech_text:
                    linke_speeches.append(speech_text)
    return linke_speeches



if __name__ == '__main__':
    # list with speaches
    speeches_linke = []
    # list with dates
    date_list =[]

    for file in os.listdir(input_folder):
        if file.endswith(".xml"):
            file_path = os.path.join(input_folder, file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            speaches = extract_speeches(root)
            if not speaches == []:
                # add date
                date = root.attrib.get("sitzung-datum", "Datum nicht gefunden")
                date_list = date_list + [date] *len(speaches)
            speeches_linke = speeches_linke + speaches

    # create array with speaches text and corresponding dates
    speeches_and_dates = np.array([speeches_linke, date_list])
    np.save(output_folder +  r"\reden_linke", speeches_and_dates)
    print(f"Found speeches of 'Die Linke': {len(speeches_linke)}")

