#
# trump_data.py
# Machine Learning Course
# Session 1  - Case Study 2
# Donald Trump QA dataset generator
#

import pandas as pd

if __name__ == "__main__":
    qa_entries = []

    # Extract trump responses and the statement that lead up the reponse
    debate_df = pd.read_csv("data/debate.csv", encoding='latin-1')
    for i in range(1, len(debate_df)):
        speaker = debate_df.loc[i, "Speaker"]
        prev_speaker = debate_df.loc[i - 1, "Speaker"]

        # Check that Trump is not responsing to himself
        if speaker == "Trump" and prev_speaker != "Trump":
            # Commit statement and response pair
            statement = debate_df.loc[i - 1, "Text"]
            response = debate_df.loc[i, "Text" ]
            
            qa_entries.append((statement, response))


    # Construct QA dataset dataframe
    n_data = len(qa_entries)
    qa_index = range(n_data)
    qa_df = pd.DataFrame(data=qa_entries,
                         index=qa_index,
                         columns=("Statement", "Response"))
                         
    # Write QA dataset to csv
    qa_df.to_csv("data/trump_qa.csv")
