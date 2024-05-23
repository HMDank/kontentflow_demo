
import gspread
from collections import Counter
from underthesea import word_tokenize


gc = gspread.service_account(filename="key.json")
sheet = gc.open_by_key("1GFTzxLWS9c7u62Hz3t6Q8ZurUUnuehO8nDB1k_sF2D0") 
wks = sheet.worksheet("DataSheet")
ana_sheet = sheet.worksheet("AnalyticsSheet")

def main():
    """
    Main function to orchestrate the process of fetching and transcribing videos.
    """
    with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file.readlines()]

    transcriptions = [item for sublist in wks.get("E2:E5") for item in sublist]

    # Tokenizing each sentence and collecting words
    all_words = []
    for sentence in transcriptions:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word not in stopwords]
        all_words.extend(filtered_words)

    # Calculating frequency of each word
    word_freq = Counter(all_words)
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Convert sorted word frequencies to a format suitable for writing to a sheet
    word_freq_data = [["Word", "Frequency"]] + [[word, count] for word, count in sorted_word_freq]

    ana_sheet.update("A1", word_freq_data)  # Adjust the range according to where you want to start writing

 

# Check if the script is run as the main program
if __name__ == "__main__":
    main()  # Execute the main function