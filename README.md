# Songs-Lyrics-Generator
This is a Natural Language Generation project that first trains itself on the corpus of lyrics by any artist and then generate its own lyrics given some starting seed lyric string.

# Dataset
The dataset was generated using a [python web scraping script](https://github.com/aakashbansal/Songs-Lyrics-Web-Scraper) for collecting lyrics of all songs by a particular artist and saved in **lyrics_scraped.txt** . Some other songs' lyrics also are manually collected from the web for which the lyrics could not be scraped and saved in **lyrics_manual_extract.txt**. Both of these files are combined into one file : **lyrics_final.txt**

# Description
The core code of the project is present in **char_level_model_eminem_lyrics_generator.ipynb** file. It reads the datset from **lyrics_final.txt**.

A lot of preprocessing is done on the dataset including removing punctuation, special characters, digits, specially formatted text ( like [intro - eminem], [beatboxing], [chorus - 50 cent], [8X] (8 times repeat), [ Verse 3 ], [ Trick Trick ], [Chorus], etc).

After the preprocessing step, the data is converted into text sequences of fixed length i.e. 30 characters ( training data - x ) and next character ( training label - y ). This is then vectorized into one hot encoding and fed to the **character level model** given in next section.

# Model
```
# Lyrics Generation Model

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(INPUT_TEXT_LEN_FIXED, len(chars))))
model.add(LSTM(64))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(chars), activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

model.summary()
```

# Training

The training was carried on for 100+ epochs with each epoch taking almost 20+ minutes.
The trained model is saved in **lyrics_gen_model.h5**.

# Results

The model can be directly run/tested using **eminem_lyrics_generate_from_seed.ipynb** file. This file loads the trained model and accepts the starting seed lyric value and then generate lyrics of its own.
```
# Characters allowed (a-z), (') , (,), (\n), (<SPACE>) - 30 chars
seed = "I've been driving around your side"

# Given function will use the seed string to generate the most probable next 400 characters
# Let's see what the model has learned!
predict_lyrics(seed)
```

These are some sample outputs that model gave for following seed lyrics :

**Note :** These are uncensored lyrics and entirely the outcome of trained model.

1. **Seed** :  "''cause it feels so empty without me"

Lyrics generated from seed:
```
''cause it feels so empty without me
i said this is what i'm sayin'
and i said this is what i do that i was back in a controversy
'cause it feels so empty without me
i said this is here in the bathroom of the back of the motherfuckers wanna start stanting
i'm tired of seen the shit to me
so fuck it, i'm sorry
i'm so bad i want me to be able to be a father
i was bad guts, what y
```

2. **Seed** : "I've been driving around your side"

Lyrics generated from seed:
```
i've been driving around your brain
see weed there's a couple of minutes
shoot and the way you don't let 'em say you ain't beautiful
oh, it's let fire
you welcome to through this shit
i'm a stall of bed it's the real life to see, i'm sorry
you don't wanna see you was a friend but i was blowing me
i said this is what i'm says
i was born and i'm not a machine got the start of me
i was black and the way that i said i'm so somethi
```

3. **Seed** :  "tired of wanting to be him"

Lyrics generated from seed :
```
 tired of wanting to be him
tired of all of my life with me
i can't even try to set she as play so much
what do i got some straight on my friends
i'm a change in the booth
i'mma be a bath of mind of the one of the game
it's a soldier
so nother song and started him
'cause i don't give a fuck
i'm murder than the preads is up in the brains, couldn't start up to be this distorth, and redict way to be the money in the denner, but i
```

# Conclusion

When the model started training itself, it knew nothing. It had no idea regarding what are lyrics, how they are constructed, what is a sentence or a word.

Over several training epochs, it learnt how characters are added together to form words, words are added to form sentences and sentences are combined to form lyrics.

The model is not producing perfectly meaningful lyrics but it still has come a long way in terms of language modelling, recognising various patterns in the lyrics and being able to generate lyrics of its own, beginning with a given seed value.

