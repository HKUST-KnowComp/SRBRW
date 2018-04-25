### Experiments

#### Sentiment classification with all users

| Methods         | Dev of Yelp Round 9 | Test of Yelp Round 9 | Dev of Yelp Round 10 | Test of Yelp Round 10 |
| --------------- | ------------------- | -------------------- | -------------------- | --------------------- |
| Word2Vec (CBOW) | 58.98               | 58.90                | 59.79                | 60.09                 |
| SWE             | 59.28               | 59.12                | 60.11                | 60.31                 |
| SRSRW           | 59.28               | 59.32                | 60.24                | 60.45                 |
| SRBRW           | 59.28               | **59.44**            | 60.32                | **60.53**             |

#### Sentiment classification with 60% users, head users and tail users

| Methods         | Overall users of Yelp Round 9 | Head users of Yelp Round 9 | Tail users of Yelp Round 9 | Overall users of Yelp Round 10 | Head users of Yelp Round 10 | Tail users of Yelp Round 10 |
| --------------- | ----------------------------- | -------------------------- | -------------------------- | ------------------------------ | --------------------------- | --------------------------- |
| Word2Vec (CBOW) | 58.90 (0.03)                  | 56.51 (0.16)               | 59.32 (0.08)               | 60.09 (0.03)                   | 57.93 (0.10)                | 60.41 (0.05)                |
| SWE             | 59.13 (0.04)                  | 57.99 (0.14)               | 59.54 (0.07)               | 60.25 (0.03)                   | 59.42 (0.10)                | 60.55 (0.05)                |
| SRSRW           | 59.38\* (0.03)                | 59.18\* (0.12)             | 59.69\* (0.06)             | 60.46\* (0.03)                 | 60.35\* (0.07)              | 60.82\* (0.03)              |
| SRBRW           | **59.43**\* (0.03)            | **59.28**\* (0.07)         | **59.72**\* (0.06)         | **60.52**\* (0.03)             | **60.47**\* (0.05)          | **60.90**\* (0.04)          |

The marker \* refers to p-value < 0.0001 in t-test compared with SWE.

#### User vectors for attention based deep learning for sentiment analysis

| Methods                               | Dev of Yelp Round 9 in HCNN | Test of Yelp Round 9 in HCNN | Dev of Yelp Round 9 in HLSTM | Test of Yelp Round 9 in HLSTM | Dev of Yelp Round 10 in HCNN | Test of Yelp Round 10 in HCNN | Dev of Yelp Round 10 in HLSTM | Test of Yelp Round 10 in HLSTM |
| ------------------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ----------------------------- | ---------------------------- | ----------------------------- | ----------------------------- | ------------------------------ |
| Word2Vec without attention            | 65.28                       | 65.22                        | 66.85                        | 66.98                         | 66.27                        | 66.19                         | 67.80                         | 67.69                          |
| Word2Vec with trained attention       | 65.89                       | 65.97                        | 66.93                        | 66.71                         | 67.04                        | 66.76                         | 67.96                         | 67.61                          |
| SWE fixed user vectors as attention   | 66.31                       | 66.39                        | 66.99                        | 66.75                         | 67.14                        | 66.93                         | 68.21                         | 67.96                          |
| SRSRW fixed user vectors as attention | 66.33                       | **66.43**                    | 67.35                        | **67.14**                     | 67.19                        | **67.07**                     | 68.23                         | 68.01                          |
| SRBRW fixed user vectors as attention | 66.33                       | 66.33                        | 67.28                        | 67.12                         | 67.28                        | 67.00                         | 68.27                         | **68.08**                      |