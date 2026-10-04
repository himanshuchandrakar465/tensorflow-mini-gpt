# Mini GPT (Word Level)

A small GPT-style transformer that reads stories and writes new ones.
It guesses the next word, again and again.

Built in Google Colab with TensorFlow.

---

## Data

- **Dataset:** TinyStories (`cfahlgren1/tinystories-gpt4-clean`)
- **Size:** 2,732,634 short stories
- **Used:** the first 1,999,999 characters
- **Vocabulary:** 4,948 unique words

---

## How the code works

### 1. Get the stories
- Download the dataset.
- Save all stories in one text file, `data/tinystories_all.txt`.
- Each story ends with `<END_STORY>`.

### 2. Clean the text
- Make all words lowercase.
- Split punctuation into its own tokens. Example: `Jack.` becomes `jack` and `.`
- The result is `token_list`.

### 3. Words to numbers
- `word_to_index`: word to number.
- `index_to_word`: number to word.

### 4. Make practice data
- `xs`: 11 words in a row (the input).
- `ys`: the next word (the answer).
- Total: 490,985 practice examples.

### 5. The transformer model
Data goes through these parts:

1. **WordAndPosition:** each word becomes 64 numbers. The spot of the word is added.
2. **Attention (4 heads):** each word looks back at earlier words.
3. **Causal mask:** a word can only look back, never forward.
4. **Add and normalize:** keeps old information and keeps numbers stable.
5. **Dense layers:** the model thinks (64, then 256, then 64 numbers).
6. **Repeat:** 2 blocks of steps 2 to 5.
7. **Last word:** keep only the last spot.
8. **Output Dense:** a score for each of the 4,948 words.

Model size: 738,964 numbers to learn.

### 6. Training
- Loss: `SparseCategoricalCrossentropy`
- Optimizer: Adam
- Batch size: 64
- Epochs: 20
- Loss went from **3.47** down to **2.15**.

### 7. Write a story
- `write_story` guesses a word, adds it, and repeats.
- `prompt` turns your text into numbers and starts the story.
- Unknown words are skipped.
- If you type fewer than 11 words, `.` is added at the start.
- If you type more than 11 words, only the last 11 are used.

Example:

```python
prompt("once there was a little boy named jack . he")
```

---

## How to run

1. Open the notebook in Google Colab.
2. Turn on the GPU (T4 works).
3. Run the cells from top to bottom.

---

## Known limits

- The model only sees the last 11 words, so stories can wander.
- It learns from 1 answer per window.
- The marker `<END_STORY>` is split into pieces by the punctuation step.
- There is no validation split yet.

---

## Ideas to improve

- Use a longer window, like 32 words.
- Use more text.
- Make `<END_STORY>` one single token.
- Add a train and validation split.
